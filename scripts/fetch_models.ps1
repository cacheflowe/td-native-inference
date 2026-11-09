# Downloads the models that aren't in the release zip into the installed models\ folder:
#
#   YOLO26 (Yolo26Detect / Yolo26Seg / Yolo26Pose)  AGPL-3.0, so not bundled; asks first. ~90 MB.
#   Kokoro-82M (MediaPipeTextToSpeech)              Apache-2.0; left out of the zip for size. ~280 MB.
#
# Every file is pinned to a Hugging Face revision and checked by SHA-256: the same files the toxes
# were tested with. Files already present are skipped. No Python needed.
#
#   powershell -ExecutionPolicy Bypass -File fetch_models.ps1 [-AcceptAgpl] [-SkipYolo] [-SkipKokoro]
param(
	[switch] $AcceptAgpl,
	[switch] $SkipYolo,
	[switch] $SkipKokoro
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'	# Invoke-WebRequest's progress bar slows downloads a lot

# In the source checkout (scripts\, next to cpp\) or installed (Documents/Derivative/Plugins/...).
# Not keyed on OnnxTOP.dll: scripts\plugins.ps1 dev moves the installed binaries out.
$checkout = Split-Path $PSScriptRoot -Parent
if (Test-Path (Join-Path $checkout 'cpp\OnnxTOP')) {
	$models = Join-Path $checkout 'models'
} else {
	$models = Join-Path $PSScriptRoot 'models'
}

$yolo = @(
	@{ path = 'onnx\yolo26\yolo26s.onnx'; size = 38239663
	   url = 'https://huggingface.co/onnx-community/yolo26s-ONNX/resolve/37669b009f416cb1df28751257d6ec5f8e4b4e20/onnx/model.onnx'
	   sha256 = 'c72bc5ad4e7f7c87666a051d0f01fc02d084688c1ca78e825031827a0590efb1' },
	@{ path = 'onnx\yolo26\yolo26s-seg.onnx'; size = 41911848
	   url = 'https://huggingface.co/zwh20081/yolo26-onnx/resolve/08680d55d85937d5b759a9926fd2b1378d413c74/yolo26s-seg.onnx'
	   sha256 = '66e12271280d46570f6afb3da87521b2392848a52ce27223dcfc801847b07c67' },
	@{ path = 'onnx\yolo26\yolo26n-pose.onnx'; size = 12081961
	   url = 'https://huggingface.co/onnx-community/yolo26n-pose-ONNX/resolve/8a9197eeec75fdb4953322a455126ca6d9780b41/onnx/model.onnx'
	   sha256 = '6ca5dc9681e0305708085715427f50779c761d5abfddaa0c38ac7c2d89b56085' }
)
$kokoro = @(
	@{ path = 'tts\kokoro_int8_fp16.litertlm'; size = 294783298
	   url = 'https://huggingface.co/litert-community/Kokoro-82M/resolve/33ad69fec6c4d9eb10a547aaf9a8c63e7c446a56/kokoro_int8_fp16.litertlm'
	   sha256 = 'e7b931ee5d4b7104012a838c9eee240d19ea69cb3ca269041a3274f83dbb7fa1' }
)

function Get-Sha256([string] $path) {
	# .NET directly: Get-FileHash can be missing when Windows PowerShell inherits PowerShell 7's module path.
	$stream = [IO.File]::OpenRead($path)
	try { return ([BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($stream)) -replace '-', '').ToLower() }
	finally { $stream.Dispose() }
}

function Get-Missing($files) { @($files | Where-Object { -not (Test-Path (Join-Path $models $_.path)) }) }

function Save-Models($files) {
	foreach ($d in $files) {
		$destination = Join-Path $models $d.path
		New-Item -ItemType Directory -Force (Split-Path $destination) | Out-Null
		Write-Host ("[fetch_models] downloading {0} ({1:N0} MB)..." -f $d.path, ($d.size / 1MB))
		$partial = "$destination.part"
		Invoke-WebRequest -Uri $d.url -OutFile $partial -UseBasicParsing
		$hash = Get-Sha256 $partial
		if ($hash -ne $d.sha256) {
			Remove-Item -Force $partial
			throw "$($d.path): checksum mismatch (got $hash). Nothing was installed; try again later."
		}
		Move-Item -Force $partial $destination
		Write-Host "[fetch_models] -> $destination"
	}
}

$missingYolo = if ($SkipYolo) { @() } else { Get-Missing $yolo }
$missingKokoro = if ($SkipKokoro) { @() } else { Get-Missing $kokoro }
if ($missingYolo.Count -eq 0 -and $missingKokoro.Count -eq 0) {
	Write-Host "[fetch_models] All models are present in $models"
	exit 0
}

if ($missingYolo.Count -gt 0) {
	Write-Host @"
YOLO26 (Ultralytics) is licensed under AGPL-3.0: https://www.ultralytics.com/license
Using it in a project you distribute or run as a network service carries AGPL obligations
unless you hold an Ultralytics Enterprise license.
"@
	$accepted = $AcceptAgpl -or ((Read-Host 'Download the YOLO26 models under these terms? [y/N]') -match '^[yY]')
	if ($accepted) { Save-Models $missingYolo } else { Write-Host '[fetch_models] Skipped YOLO26.' }
}
if ($missingKokoro.Count -gt 0) {
	Write-Host '[fetch_models] Kokoro-82M text to speech (Apache-2.0).'
	Save-Models $missingKokoro
}
Write-Host '[fetch_models] Done. The first run of each YOLO model builds a TensorRT engine (a few minutes).'
