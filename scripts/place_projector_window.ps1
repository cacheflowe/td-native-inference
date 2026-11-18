<#
.SYNOPSIS
    Moves a TouchDesigner Window COMP onto a monitor, full screen and above the taskbar.

.DESCRIPTION
    TD's Window COMP positions by its own offsets, which skew when the monitors have different Windows
    display scaling (a 150% primary next to a 100% projector put a window meant for x=3840 at x=5120), and
    the Window COMP's display index changes when Windows renumbers the displays (after the projector is
    moved or replugged). This places the window with Windows' own API instead: exactly the monitor's
    rectangle, topmost so the taskbar doesn't show over it.

    Run it after the window opens (TD restarts close it; open it again from the demo first). Set the Window
    COMP to a size of 3840x2160 or larger and feed it a TOP at that resolution (ProjectionRehearsal has a
    projector_scale upscale in front of its window), because Custom size shows the TOP at native pixels.

    .\place_projector_window.ps1                      # the first non-primary monitor
    .\place_projector_window.ps1 -Monitor '\\.\DISPLAY1'
#>
param(
	[string]$Title = '/projector',		# the window's title ends with the COMP's path
	[string]$Monitor = '',				# a device name from the list below; default: the first non-primary one
	[switch]$NoTopmost
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type @'
using System; using System.Text; using System.Runtime.InteropServices;
public class WinPlace {
	[DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc f, IntPtr p);
	[DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
	[DllImport("user32.dll")] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
	[DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int cx, int cy, uint flags);
	public delegate bool EnumProc(IntPtr h, IntPtr p);
	public static IntPtr Find(string suffix) {
		IntPtr found = IntPtr.Zero;
		EnumWindows((h, p) => { var s = new StringBuilder(300); GetWindowText(h, s, 300); if (IsWindowVisible(h) && s.ToString().EndsWith(suffix)) found = h; return true; }, IntPtr.Zero);
		return found;
	}
}
'@

$screens = [Windows.Forms.Screen]::AllScreens
$screens | ForEach-Object { Write-Host ('{0}{1} {2}x{3} at {4},{5}' -f $_.DeviceName, $(if ($_.Primary) { ' (primary)' } else { '' }), $_.Bounds.Width, $_.Bounds.Height, $_.Bounds.X, $_.Bounds.Y) }
$target = if ($Monitor) { $screens | Where-Object { $_.DeviceName -eq $Monitor } | Select-Object -First 1 } else { $screens | Where-Object { -not $_.Primary } | Select-Object -First 1 }
if (-not $target) { Write-Error 'No such monitor.'; exit 1 }
$window = [WinPlace]::Find($Title)
if ($window -eq [IntPtr]::Zero) { Write-Error "No visible window whose title ends with '$Title'. Open the Window COMP first."; exit 1 }
$after = if ($NoTopmost) { [IntPtr]::Zero } else { [IntPtr](-1) }	# HWND_TOPMOST
[WinPlace]::SetWindowPos($window, $after, $target.Bounds.X, $target.Bounds.Y, $target.Bounds.Width, $target.Bounds.Height, 0x40) | Out-Null	# SWP_SHOWWINDOW
Write-Host ('Placed on {0}: {1}x{2} at {3},{4}' -f $target.DeviceName, $target.Bounds.Width, $target.Bounds.Height, $target.Bounds.X, $target.Bounds.Y)
