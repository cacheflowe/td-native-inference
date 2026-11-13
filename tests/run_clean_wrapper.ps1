# Clean-process test of an exported wrapper tox (driver: tests/td_clean_wrapper.py).
#
# Installs the plugin folder temporarily where users install it,
# Documents\Derivative\Plugins\td-native-inference\, launches a fresh TouchDesigner on the driver
# tox in -CaseDir (no plugins or models anywhere near it), loads each -Wrapper tox in turn, feeds
# it the matching -Image (one image = used for all) and prints the JSON report. Afterwards the
# temporary install is removed and Plugins.json restored. TD asks once to allow each plugin DLL
# in the new location.
#
#   -Package <extracted release folder>  test the release: its td-native-inference\ and tox\
#   (default)                            test this checkout: plugins\ + the models in -Models
#   -NoInstall                           don't install plugins (tests the missing-plugin case)
param(
	[Parameter(Mandatory)] [string] $CaseDir,
	[string[]] $Wrapper = @('MediaPipeHands'),
	[string[]] $Image = @('build/thumbs_up.jpg'),
	[string] $Package,
	[string[]] $Models = @('models/mediapipe/gesture_recognizer.task'),
	[switch] $NoInstall,
	[int] $TimeoutSeconds = 180
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$td = 'C:\Program Files\Derivative\TouchDesigner\bin\TouchDesigner.exe'
$plugins = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'Derivative\Plugins'
$install = Join-Path $plugins 'td-native-inference'
$allowList = Join-Path $plugins 'Plugins.json'
$toxDir = if ($Package) { Join-Path $Package 'tox' } else { Join-Path $repo 'tox' }

if (Test-Path $CaseDir) { Remove-Item -Recurse -Force $CaseDir }
New-Item -ItemType Directory -Force "$CaseDir\package" | Out-Null
Copy-Item "$repo\build\clean-wrapper\RunCleanWrapper.tox" $CaseDir
$cases = @()
for ($i = 0; $i -lt $Wrapper.Count; $i++) {
	Copy-Item (Join-Path $toxDir "$($Wrapper[$i]).tox") "$CaseDir\package\"
	$cases += @{ tox = "$CaseDir\package\$($Wrapper[$i]).tox"; image = (Join-Path $repo $Image[[Math]::Min($i, $Image.Count - 1)]) }
}

$installed = $false
$process = $null
$savedAllowList = if (Test-Path $allowList) { Get-Content -Raw $allowList } else { $null }
try {
	if (-not $NoInstall) {
		if (Test-Path $install) { throw "$install already exists; remove it first (it would mask this test)." }
		$installed = $true
		if ($Package) {
			Copy-Item -Recurse (Join-Path $Package 'td-native-inference') $install
		} else {
			New-Item -ItemType Directory -Force $install | Out-Null
			foreach ($file in 'OnnxTOP.dll', 'onnxtop_tool.exe', 'MediaPipeTOP.dll', 'MediaPipeCHOP.dll', 'TrackerDAT.dll', 'PuppetCHOP.dll', 'libmediapipe.dll') {
				Copy-Item "$repo\plugins\$file" $install
			}
			foreach ($model in $Models) {
				$target = Join-Path $install $model
				New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
				Copy-Item (Join-Path $repo $model) $target
			}
		}
	}
	$result = "$CaseDir\result.json"
	$env:TDNI_CASES = ConvertTo-Json -Compress -InputObject @($cases)
	$env:TDNI_RESULT = $result
	$process = Start-Process $td -ArgumentList "`"$CaseDir\RunCleanWrapper.tox`"" -WorkingDirectory $CaseDir -PassThru
	$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
	while (-not (Test-Path $result) -and -not $process.HasExited -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 500 }
	if (Test-Path $result) { Get-Content $result }
	else {
		"[CleanWrapper] no result (exited=$($process.HasExited)); an unanswered Allow prompt is the usual cause"
		if (Test-Path "$result.partial") { "[CleanWrapper] finished cases:"; Get-Content "$result.partial" }
		else { "[CleanWrapper] no case finished (stuck before or in the first case)" }
	}
	if (-not $process.HasExited) { $process.WaitForExit(15000) | Out-Null }
} finally {
	# The test's own TD (a timeout, or stuck on a dialog) would keep the install's DLLs locked.
	if ($process -and -not $process.HasExited) { Stop-Process -Id $process.Id -Force; Start-Sleep -Seconds 2 }
	# Restore the allow-list first, so a failed removal below can't leave it changed.
	if ($null -ne $savedAllowList) { Set-Content -NoNewline -Path $allowList -Value $savedAllowList }
	if ($installed) {
		# A TensorRT engine build (onnxtop_tool.exe) may still be writing into the temporary install.
		Get-Process onnxtop_tool -ErrorAction SilentlyContinue |
			Where-Object { $_.Path -and $_.Path.StartsWith($install, [StringComparison]::OrdinalIgnoreCase) } |
			Stop-Process -Force
		Start-Sleep -Seconds 1
		try { Remove-Item -Recurse -Force $install -ErrorAction Stop }
		catch {
			# Another TD has its DLLs loaded: they can't be deleted but can be moved on the same drive,
			# so get the folder out of the Plugins path before a TD started later loads it.
			$parking = Join-Path $env:TEMP "td-native-inference-leftover-$(Get-Date -Format yyyyMMddHHmmss)"
			Move-Item $install $parking
			Write-Warning "Could not remove $install (another TouchDesigner has its DLLs loaded). Moved it to $parking; delete that after the TD closes."
		}
	}
}
