@echo off
rem Builds the puppet core, PuppetCHOP.dll (copied to plugins) and puppet_replay.dll (the tests' engine).
rem Same toolchain as the other plugins (MSVC v142); no CUDA. Override VCVARS / CMAKE_NINJA_ROOT.
setlocal
if "%VCVARS%"=="" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if "%CMAKE_NINJA_ROOT%"=="" set "CMAKE_NINJA_ROOT=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake"
if "%CONFIG%"=="" set "CONFIG=Release"

call "%VCVARS%" >nul || exit /b 1
set "PATH=%CMAKE_NINJA_ROOT%\Ninja;%CMAKE_NINJA_ROOT%\CMake\bin;%PATH%"
cd /d "%~dp0"

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=%CONFIG% || exit /b 1
cmake --build build || exit /b 1
