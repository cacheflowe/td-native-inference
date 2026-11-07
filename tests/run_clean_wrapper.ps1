# Clean-process test of an exported wrapper tox (see tests/td_clean_wrapper.py).
#   -Install: copy the plugins + the listed models into <CaseDir>/Plugins/td-native-inference/,
#             the layout a release user gets in Documents/Derivative/Plugins/. Without it the
#             fresh project has no plugins at all.
# TD asks once to allow each plugin DLL in a new location.
param(
	[Parameter(Mandatory)] [string] $CaseDir,
	[string] $Wrapper = 'MediaPipeHands',
	[string] $Image = 'build/thumbs_up.jpg',
	[string[]] $Models = @('models/mediapipe/gesture_recognizer.task'),
	[switch] $Install,
	[int] $TimeoutSeconds = 180
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$td = 'C:\Program Files\Derivative\TouchDesigner\bin\TouchDesigner.exe'

if (Test-Path $CaseDir) { Remove-Item -Recurse -Force $CaseDir }
New-Item -ItemType Directory -Force "$CaseDir\package" | Out-Null
Copy-Item "$repo\build\clean-wrapper\RunCleanWrapper.tox" $CaseDir
Copy-Item "$repo\tox\$Wrapper.tox" "$CaseDir\package\"
if ($Install) {
	$pluginFolder = "$CaseDir\Plugins\td-native-inference"
	New-Item -ItemType Directory -Force $pluginFolder | Out-Null
	foreach ($file in 'OnnxTOP.dll', 'onnxtop_tool.exe', 'MediaPipeTOP.dll', 'libmediapipe.dll') {
		Copy-Item "$repo\plugins\$file" $pluginFolder
	}
	foreach ($model in $Models) {
		$target = Join-Path $pluginFolder $model
		New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
		Copy-Item (Join-Path $repo $model) $target
	}
}

$result = "$CaseDir\result.json"
$env:TDNI_WRAPPER_TOX = "$CaseDir\package\$Wrapper.tox"
$env:TDNI_IMAGE = Join-Path $repo $Image
$env:TDNI_RESULT = $result
$process = Start-Process $td -ArgumentList "`"$CaseDir\RunCleanWrapper.tox`"" -WorkingDirectory $CaseDir -PassThru
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while (-not (Test-Path $result) -and -not $process.HasExited -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 500 }
if (Test-Path $result) { Get-Content $result } else { "[CleanWrapper] no result (exited=$($process.HasExited))" }
