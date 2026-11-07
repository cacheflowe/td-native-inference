# Exports the YOLO26 models used by Yolo26Detect / Yolo26Seg / Yolo26Pose into models/onnx/yolo26/.
#
# YOLO26 is licensed by Ultralytics under AGPL-3.0, so it isn't bundled in the release: this
# script exports it on your machine with the official Ultralytics package (which downloads
# Ultralytics' own weights). Every other model ships in the release.
#
# Needs Python 3.9-3.12 (python.org or conda). Creates a temporary virtual environment
# (CPU PyTorch + ultralytics, about 1 GB) and removes it afterwards unless -KeepEnv.
#
#   powershell -ExecutionPolicy Bypass -File fetch_models.ps1
#   ... -Python "C:\path\to\python.exe" -AcceptAgpl -KeepEnv
param(
	[string] $Python = 'python',
	[switch] $AcceptAgpl,
	[switch] $KeepEnv
)
$ErrorActionPreference = 'Stop'

# Installed (next to OnnxTOP.dll in Documents/Derivative/Plugins/...) or in the source checkout.
if (Test-Path (Join-Path $PSScriptRoot 'OnnxTOP.dll')) {
	$models = Join-Path $PSScriptRoot 'models'
} else {
	$models = Join-Path (Split-Path $PSScriptRoot -Parent) 'models'
}
$target = Join-Path $models 'onnx\yolo26'
# Exports: Ultralytics weights name -> file name the wrappers expect.
$exports = [ordered]@{
	'yolo26s.pt' = 'yolo26s.onnx'
	'yolo26s-seg.pt' = 'yolo26s-seg.onnx'
	'yolo26n-pose.pt' = 'yolo26n-pose.onnx'
}

$missing = @($exports.Values | Where-Object { -not (Test-Path (Join-Path $target $_)) })
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
	$answer = Read-Host 'Export the YOLO26 models under these terms? [y/N]'
	if ($answer -notmatch '^[yY]') { Write-Host '[fetch_models] Cancelled.'; exit 1 }
}

$work = Join-Path ([IO.Path]::GetTempPath()) 'td-native-inference-yolo'
$venv = Join-Path $work 'venv'
New-Item -ItemType Directory -Force $work, $target | Out-Null
$venvPython = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
	Write-Host '[fetch_models] Creating a temporary Python environment...'
	& $Python -m venv $venv
	if ($LASTEXITCODE -ne 0) { throw "Could not create a virtual environment with '$Python'. Pass -Python <path to python.exe>." }
	& $venvPython -m pip install -q --upgrade pip
	& $venvPython -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision
	& $venvPython -m pip install -q ultralytics onnx onnxslim
	if ($LASTEXITCODE -ne 0) { throw 'pip install failed (see above).' }
}

Push-Location $work
try {
	foreach ($weights in $exports.Keys) {
		$name = $exports[$weights]
		$destination = Join-Path $target $name
		if (Test-Path $destination) { Write-Host "[fetch_models] have $name"; continue }
		Write-Host "[fetch_models] exporting $name..."
		# The settings the wrappers were tested with: NMS-free end2end head, fixed 640x640, opset 12.
		$code = "from ultralytics import YOLO; print(YOLO('$weights').export(format='onnx', opset=12, simplify=True, dynamic=False))"
		$exported = (& $venvPython -c $code | Select-Object -Last 1).Trim()
		if ($LASTEXITCODE -ne 0 -or -not (Test-Path $exported)) { throw "Export of $weights failed." }
		Move-Item -Force $exported $destination
		Write-Host "[fetch_models] -> $destination"
	}
} finally {
	Pop-Location
	if (-not $KeepEnv) { Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue }
}
Write-Host '[fetch_models] Done. The first run of each model builds a TensorRT engine (a few minutes).'
