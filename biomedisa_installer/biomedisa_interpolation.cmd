@echo off

REM remember current working directory
set OLDDIR=%CD%

REM change working directory
cd %~dp0

REM get version
FOR /F "tokens=* delims=" %%v in (version.txt) DO (set VERSION=%%v)

REM change working directory
cd %USERPROFILE%

REM run biomedisa interpolation
wsl -d %VERSION% -u biomedisa AppData/%VERSION%/biomedisa_interpolation.sh %VERSION% %*

REM change working directory
cd %OLDDIR%
