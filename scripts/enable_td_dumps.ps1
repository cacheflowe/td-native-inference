<#
.SYNOPSIS
    Registers Windows Error Reporting to write full crash dumps of TouchDesigner.exe into build\dumps.

.DESCRIPTION
    TouchDesigner's own TouchDesignerCrash*.dmp does not appear for every crash (the Nov 2026 ones left
    nothing). With this on, Windows writes a full dump whenever TouchDesigner.exe crashes, which
    cdb / WinDbg can read: cdb -z <dump> -c "!analyze -v; k; q". See docs/learnings/td-crashes.md.

    Needs an elevated PowerShell (it writes under HKLM). Run it once; -Disable removes it.
    Dumps are large (the process's whole memory, a few GB), so DumpCount keeps only the last 3.
#>
param([switch]$Disable)

$key = 'HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\TouchDesigner.exe'
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
	Write-Error 'Run this from an elevated (administrator) PowerShell.'
	exit 1
}

if ($Disable) {
	Remove-Item $key -ErrorAction SilentlyContinue
	Write-Host 'Crash dumps for TouchDesigner.exe turned off.'
	exit 0
}

$folder = Join-Path (Split-Path $PSScriptRoot -Parent) 'build\dumps'
New-Item -ItemType Directory -Force $folder | Out-Null
New-Item -Path $key -Force | Out-Null
New-ItemProperty -Path $key -Name DumpFolder -Value $folder -PropertyType ExpandString -Force | Out-Null
New-ItemProperty -Path $key -Name DumpType -Value 2 -PropertyType DWord -Force | Out-Null	# 2 = full
New-ItemProperty -Path $key -Name DumpCount -Value 3 -PropertyType DWord -Force | Out-Null
Write-Host "Full crash dumps of TouchDesigner.exe go to $folder (last 3 kept)."
