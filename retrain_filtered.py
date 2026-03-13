import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import BASE_MODEL_NAME, DATA_PROCESSED, MAX_LENGTH, MODELS_DIR, SEED

LABEL2ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class ABSADataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def load_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"review_id", "aspect", "review_text", "weak_label"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan pada clean data: {sorted(missing)}")

    work = df.copy()
    work = work[work["weak_label"].isin(LABEL2ID.keys())].copy()
    work["task_text"] = "[ASPECT=" + work["aspect"].astype(str) + "] " + work[
        "review_text"
    ].astype(str)
    work["label_id"] = work["weak_label"].map(LABEL2ID)
    work = work.dropna(subset=["task_text", "label_id"]).reset_index(drop=True)
    work["label_id"] = work["label_id"].astype(int)
    return work


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Retrain IndoBERT baseline with uncertainty-filtered clean data"
    )
    parser.add_argument("--clean_csv", default=str(DATA_PROCESSED / "noise" / "clean_data.csv"))
    parser.add_argument("--model_name", default=BASE_MODEL_NAME)
    parser.add_argument("--output_dir", default=str(MODELS_DIR / "retrained"))
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_clean_data(args.clean_csv)

    if len(data) < 30:
        raise ValueError("Data clean terlalu sedikit untuk retraining.")

    train_df, test_df = train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=data["label_id"],
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=train_df["label_id"],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_enc = tokenizer(
        train_df["task_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=args.max_length,
    )
    val_enc = tokenizer(
        val_df["task_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=args.max_length,
    )
    test_enc = tokenizer(
        test_df["task_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=args.max_length,
    )

    train_dataset = ABSADataset(train_enc, train_df["label_id"].tolist())
    val_dataset = ABSADataset(val_enc, val_df["label_id"].tolist())
    test_dataset = ABSADataset(test_enc, test_df["label_id"].tolist())

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time

    test_output = trainer.predict(test_dataset)
    test_logits = test_output.predictions
    y_true = np.array(test_df["label_id"].tolist())
    y_pred = np.argmax(test_logits, axis=1)

    metrics = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "test_f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_total_clean": int(len(data)),
        "training_time_seconds": round(train_time, 2),
        "label_distribution": data["weak_label"].value_counts().to_dict(),
    }

    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=[ID2LABEL[i] for i in range(3)],
        digits=4,
        zero_division=0,
    )

    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))

    predictions_df = test_df[["review_id", "aspect", "review_text", "weak_label"]].copy()
    predictions_df["pred_label"] = [ID2LABEL[int(x)] for x in y_pred]
    predictions_df["is_error"] = predictions_df["weak_label"] != predictions_df["pred_label"]
    predictions_df.to_csv(output_dir / "filtered_test_predictions.csv", index=False)
    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)

    with open(output_dir / "filtered_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(output_dir / "filtered_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("[RETRAIN] Training clean-data selesai.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
