# Downloads the YOLO26 models used by Yolo26Detect / Yolo26Seg / Yolo26Pose into models/onnx/yolo26/.
#
# YOLO26 is licensed by Ultralytics under AGPL-3.0, so it isn't bundled in the release (every other
# model is). These are ONNX exports of Ultralytics' weights published on Hugging Face, pinned to a
# revision and checked by SHA-256: the same files the wrappers were tested with. About 90 MB, no
# Python needed.
#
#   powershell -ExecutionPolicy Bypass -File fetch_models.ps1 [-AcceptAgpl]
param(
	[switch] $AcceptAgpl
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'	# Invoke-WebRequest's progress bar slows downloads a lot

# Installed (next to OnnxTOP.dll in Documents/Derivative/Plugins/...) or in the source checkout.
if (Test-Path (Join-Path $PSScriptRoot 'OnnxTOP.dll')) {
	$models = Join-Path $PSScriptRoot 'models'
} else {
	$models = Join-Path (Split-Path $PSScriptRoot -Parent) 'models'
}
$target = Join-Path $models 'onnx\yolo26'

$downloads = @(
	@{ name = 'yolo26s.onnx'; size = 38239663
	   url = 'https://huggingface.co/onnx-community/yolo26s-ONNX/resolve/37669b009f416cb1df28751257d6ec5f8e4b4e20/onnx/model.onnx'
	   sha256 = 'c72bc5ad4e7f7c87666a051d0f01fc02d084688c1ca78e825031827a0590efb1' },
	@{ name = 'yolo26s-seg.onnx'; size = 41911848
	   url = 'https://huggingface.co/zwh20081/yolo26-onnx/resolve/08680d55d85937d5b759a9926fd2b1378d413c74/yolo26s-seg.onnx'
	   sha256 = '66e12271280d46570f6afb3da87521b2392848a52ce27223dcfc801847b07c67' },
	@{ name = 'yolo26n-pose.onnx'; size = 12081961
	   url = 'https://huggingface.co/onnx-community/yolo26n-pose-ONNX/resolve/8a9197eeec75fdb4953322a455126ca6d9780b41/onnx/model.onnx'
	   sha256 = '6ca5dc9681e0305708085715427f50779c761d5abfddaa0c38ac7c2d89b56085' }
)

$missing = @($downloads | Where-Object { -not (Test-Path (Join-Path $target $_.name)) })
if ($missing.Count -eq 0) {
	Write-Host "[fetch_models] All YOLO26 models are present in $target"
	exit 0
}

Write-Host @"
YOLO26 (Ultralytics) is licensed under AGPL-3.0: https://www.ultralytics.com/license
Using it in a project you distribute or run as a network service carries AGPL obligations
unless you hold an Ultralytics Enterprise license.
"@
if (-not $AcceptAgpl) {
	$answer = Read-Host 'Download the YOLO26 models under these terms? [y/N]'
	if ($answer -notmatch '^[yY]') { Write-Host '[fetch_models] Cancelled.'; exit 1 }
}
New-Item -ItemType Directory -Force $target | Out-Null

foreach ($d in $missing) {
	$destination = Join-Path $target $d.name
	Write-Host ("[fetch_models] downloading {0} ({1:N0} MB)..." -f $d.name, ($d.size / 1MB))
	$partial = "$destination.part"
	Invoke-WebRequest -Uri $d.url -OutFile $partial -UseBasicParsing
	$hash = (Get-FileHash $partial -Algorithm SHA256).Hash.ToLower()
	if ($hash -ne $d.sha256) {
		Remove-Item -Force $partial
		throw "$($d.name): checksum mismatch (got $hash). Nothing was installed; try again later."
	}
	Move-Item -Force $partial $destination
	Write-Host "[fetch_models] -> $destination"
}
Write-Host '[fetch_models] Done. The first run of each model builds a TensorRT engine (a few minutes).'
