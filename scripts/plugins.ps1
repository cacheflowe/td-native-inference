# Switches which td-native-inference plugins TouchDesigner loads on this machine. With an install in
# Documents\Derivative\Plugins\ AND the dev project's own plugins\ folder, TD registers every custom
# OP twice and either copy may serve a node, so only one should hold the binaries at a time.
#
#   dev      Move the installed binaries out of Plugins\ (to ..\Derivative\td-native-inference.disabled\)
#            so the dev project loads its own plugins\ build.
#   install  Copy the local plugins\ build into the install folder, for use in every project.
#   status   Show which binaries are where.
#
# Only the compiled files move. The install's models\ folder belongs to fetch_models.ps1 and the
# release zip, and stays put. TouchDesigner must be closed (it locks loaded DLLs).
#
#   powershell -ExecutionPolicy Bypass -File scripts\plugins.ps1 dev|install|status
param(
	[Parameter(Mandatory = $true)]
	[ValidateSet('dev', 'install', 'status')]
	[string] $Mode
)
$ErrorActionPreference = 'Stop'

# Same binaries as the release zip (scripts/release.py).
$binaries = @('OnnxTOP.dll', 'onnxtop_tool.exe', 'MediaPipeTOP.dll', 'MediaPipeCHOP.dll', 'TrackerDAT.dll', 'libmediapipe.dll')

$build = Join-Path (Split-Path $PSScriptRoot -Parent) 'plugins'
$derivative = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'Derivative'
$install = Join-Path $derivative 'Plugins\td-native-inference'
$stash = Join-Path $derivative 'td-native-inference.disabled'

function Show-Folder([string] $label, [string] $dir) {
	$present = @($binaries | Where-Object { Test-Path (Join-Path $dir $_) })
	if ($present.Count -eq 0) {
		Write-Host "[plugins] $label ($dir): none"
		return
	}
	$newest = ($present | ForEach-Object { (Get-Item (Join-Path $dir $_)).LastWriteTime } | Measure-Object -Maximum).Maximum
	Write-Host "[plugins] $label ($dir): $($present.Count)/$($binaries.Count), newest $($newest.ToString('yyyy-MM-dd HH:mm'))"
}

function Assert-TdClosed {
	if (Get-Process -Name 'TouchDesigner*' -ErrorAction SilentlyContinue) {
		throw 'TouchDesigner is running; close it first (it locks the plugin DLLs).'
	}
}

switch ($Mode) {
	'status' {
		Show-Folder 'installed' $install
		Show-Folder 'disabled ' $stash
		Show-Folder 'local    ' $build
		$installed = @($binaries | Where-Object { Test-Path (Join-Path $install $_) }).Count
		if ($installed -gt 0) {
			Write-Host '[plugins] The dev .toe loads two copies of each OP while both installed and local exist.'
		}
	}
	'dev' {
		Assert-TdClosed
		$moved = 0
		New-Item -ItemType Directory -Force $stash | Out-Null
		foreach ($name in $binaries) {
			$src = Join-Path $install $name
			if (Test-Path $src) {
				Move-Item -Force $src (Join-Path $stash $name)
				$moved++
			}
		}
		Write-Host "[plugins] Moved $moved binaries to $stash. TD now loads only the project's plugins\ build."
	}
	'install' {
		Assert-TdClosed
		$missing = @($binaries | Where-Object { -not (Test-Path (Join-Path $build $_)) })
		if ($missing.Count -gt 0) {
			throw "Local build is missing: $($missing -join ', ') (in $build)"
		}
		New-Item -ItemType Directory -Force $install | Out-Null
		foreach ($name in $binaries) {
			Copy-Item -Force (Join-Path $build $name) (Join-Path $install $name)
		}
		Write-Host "[plugins] Installed the local build into $install."
		Write-Host '[plugins] TD will ask to allow the new DLLs on first launch. Open the dev .toe and it loads two copies again: run "dev" first.'
	}
}
