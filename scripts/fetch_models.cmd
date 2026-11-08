@echo off
rem Double-click to download the models that aren't in the zip (YOLO26, the Kokoro voice).
rem Runs fetch_models.ps1 with a one-off execution policy, because Windows refuses to run an unsigned
rem script that came out of a downloaded zip. Any arguments are passed on (-SkipYolo, -SkipKokoro).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0fetch_models.ps1" %*
echo.
pause
