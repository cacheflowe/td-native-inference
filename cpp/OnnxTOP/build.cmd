@echo off
rem Builds OnnxTOP.dll (+ ort_bench.exe) with Ninja.
rem
rem Compiler: MSVC v142 (VS 2019 toolset, same as TD's own CudaTOP sample). CUDA 12.3 can't
rem use newer MSVC: 14.4x's STL hard-errors on CUDA < 12.4 (STL1002). With CUDA >= 12.4
rem installed you can point VCVARS at VS 2022 instead.
rem
rem Override any of these before calling if your installs live elsewhere.
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
