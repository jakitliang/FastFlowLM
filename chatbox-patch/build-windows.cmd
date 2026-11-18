@echo off
setlocal
rem No driver installation, backend changes, or model downloads.
if /i not "%VSCMD_ARG_TGT_ARCH%"=="x64" (
	echo Run this script from an x64 Native Tools Command Prompt for VS 2022.
	exit /b 1
)
set "CMAKE_BUILD_PARALLEL_LEVEL=8"
set "CARGO_BUILD_JOBS=4"
rem Sources are UTF-8; the host's default Chinese code page misparses comments.
set "CL=/utf-8 %CL%"
cd /d "%~dp0..\src"
if errorlevel 1 exit /b 1
cmake -S test/model_metadata -B build-tests/model_metadata -G Ninja -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build build-tests/model_metadata --parallel 8
if errorlevel 1 exit /b 1
ctest --test-dir build-tests/model_metadata --output-on-failure
if errorlevel 1 exit /b 1
cmake -S test/http_guard -B build-tests/http_guard -G Ninja -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build build-tests/http_guard --parallel 8
if errorlevel 1 exit /b 1
ctest --test-dir build-tests/http_guard --output-on-failure
if errorlevel 1 exit /b 1
cmake --preset windows-ninja -DFLM_ENABLE_OGA_HIPEP=OFF -DFLM_BUILD_GEMMA4E=OFF
if errorlevel 1 exit /b 1
cmake --build build --parallel 8
exit /b %errorlevel%