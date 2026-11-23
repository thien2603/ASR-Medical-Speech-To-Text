$ErrorActionPreference = "Stop"

$runtime = if ($env:ASR_RUNTIME_DIR) { $env:ASR_RUNTIME_DIR } else { "E:\SpeechToText_Medical_runtime" }
$python = Join-Path $runtime ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Không tìm thấy môi trường Python tại $python"
}

$env:TEMP = Join-Path $runtime "temp"
$env:TMP = $env:TEMP
$env:HF_HOME = Join-Path $runtime "cache\huggingface"
$env:HF_DATASETS_CACHE = Join-Path $runtime "cache\datasets"
$env:NEMO_CACHE_DIR = Join-Path $runtime "cache\nemo"
$env:WANDB_MODE = "disabled"
$env:ASR_DEVICE = "cpu"
$env:GRADIO_ANALYTICS_ENABLED = "False"

Push-Location -LiteralPath $PSScriptRoot
$asrExitCode = 0
try {
    & $python -u -m inference.app
    $asrExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $asrExitCode
