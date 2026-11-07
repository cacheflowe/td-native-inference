@echo off
rem Builds MediaPipeTOP.dll with Ninja (same toolchain as cpp\OnnxTOP\build.cmd: MSVC v142 + CUDA 12.3).
rem Run fetch_mediapipe.ps1 once first to install libmediapipe.dll and the .task models.
setlocal
if "%VCVARS%"=="" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if "%CMAKE_NINJA_ROOT%"=="" set "CMAKE_NINJA_ROOT=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake"
if "%CUDA_ROOT%"=="" set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3"
if "%CONFIG%"=="" set "CONFIG=Release"

call "%VCVARS%" >nul || exit /b 1
set "PATH=%CMAKE_NINJA_ROOT%\Ninja;%CMAKE_NINJA_ROOT%\CMake\bin;%CUDA_ROOT%\bin;%PATH%"
cd /d "%~dp0"

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=%CONFIG% ^
	"-DCMAKE_CUDA_COMPILER=%CUDA_ROOT%\bin\nvcc.exe" || exit /b 1
cmake --build build || exit /b 1
