$ErrorActionPreference = 'Stop'

$python = ".\.venv\Scripts\python.exe"
$inputCsv = "data/processed/dataset_absa_50k_v2_intersection.csv"
$modelName = "indobenchmark/indobert-base-p1"
$maxLength = 128
$testSize = 0.2
$valSize = 0.1
$seed = 42
$epochsList = @(3, 5, 8)

$baselineBatchSize = 8
$baselineLearningRate = 2e-5

$loraBatchSize = 16
$loraLearningRate = 2e-4
$loraR = 16
$loraAlpha = 32
$loraDropout = 0.1

$results = @()

function Invoke-TrainingRun {
    param(
        [string]$Label,
        [string]$OutputDir,
        [string[]]$Arguments
    )

    Write-Host "[$Label] Running -> $OutputDir"
    & $python @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "$Label run failed for output dir: $OutputDir"
    }
}

function Read-Metrics {
    param(
        [string]$ModelType,
        [int]$Epochs,
        [string]$OutputDir
    )

    $metricsPath = Join-Path $OutputDir "metrics.json"
    if (-not (Test-Path $metricsPath)) {
        throw "Metrics file not found: $metricsPath"
    }

    $metrics = Get-Content $metricsPath -Raw | ConvertFrom-Json
    return [PSCustomObject]@{
        model = $ModelType
        epochs = $Epochs
        accuracy = [double]$metrics.test_accuracy
        f1_macro = [double]$metrics.test_f1_macro
        f1_weighted = [double]$metrics.test_f1_weighted
        training_time_seconds = if ($null -ne $metrics.training_time_seconds) { [double]$metrics.training_time_seconds } else { $null }
        trainable_params = if ($null -ne $metrics.trainable_params) { [double]$metrics.trainable_params } else { $null }
        output_dir = $OutputDir
    }
}

foreach ($epochs in $epochsList) {
    $baselineOutputDir = "models/baseline/epoch_$epochs"
    Invoke-TrainingRun `
        -Label "BASELINE" `
        -OutputDir $baselineOutputDir `
        -Arguments @(
            "train_baseline.py",
            "--input_csv", $inputCsv,
            "--model_name", $modelName,
            "--output_dir", $baselineOutputDir,
            "--max_length", $maxLength,
            "--test_size", $testSize,
            "--val_size", $valSize,
            "--epochs", $epochs,
            "--batch_size", $baselineBatchSize,
            "--lr", $baselineLearningRate,
            "--seed", $seed
        )
    $results += Read-Metrics -ModelType "baseline" -Epochs $epochs -OutputDir $baselineOutputDir

    $loraOutputDir = "models/lora/epoch_$epochs"
    Invoke-TrainingRun `
        -Label "LORA" `
        -OutputDir $loraOutputDir `
        -Arguments @(
            "train_lora.py",
            "--input_csv", $inputCsv,
            "--model_name", $modelName,
            "--output_dir", $loraOutputDir,
            "--max_length", $maxLength,
            "--test_size", $testSize,
            "--val_size", $valSize,
            "--epochs", $epochs,
            "--batch_size", $loraBatchSize,
            "--lr", $loraLearningRate,
            "--seed", $seed,
            "--lora_r", $loraR,
            "--lora_alpha", $loraAlpha,
            "--lora_dropout", $loraDropout
        )
    $results += Read-Metrics -ModelType "lora" -Epochs $epochs -OutputDir $loraOutputDir
}

Write-Host ""
Write-Host "[RUNNER] Experiment summary"
$results |
    Sort-Object model, epochs |
    Select-Object `
        model,
        epochs,
        @{Name = "accuracy"; Expression = { "{0:N4}" -f $_.accuracy } },
        @{Name = "f1_macro"; Expression = { "{0:N4}" -f $_.f1_macro } },
        @{Name = "f1_weighted"; Expression = { "{0:N4}" -f $_.f1_weighted } },
        @{Name = "time_s"; Expression = { if ($null -ne $_.training_time_seconds) { "{0:N2}" -f $_.training_time_seconds } else { "N/A" } } },
        @{Name = "trainable_params"; Expression = { if ($null -ne $_.trainable_params) { "{0:N0}" -f $_.trainable_params } else { "N/A" } } },
        output_dir |
    Format-Table -AutoSize

Write-Host ""
Write-Host "[RUNNER] All baseline and LoRA experiments for epochs 3, 5, and 8 completed successfully."
