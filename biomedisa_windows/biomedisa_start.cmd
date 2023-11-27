@echo off

REM get version
FOR /F "tokens=* delims=" %%v in (version.txt) DO (set VERSION=%%v)

REM get current date
for /f "tokens=1,2 delims==" %%i in ('wmic os get LocalDateTime /VALUE') do (if %%i EQU LocalDateTime set ldt=%%j)
set CURRENT_DATE=%ldt:~0,4%%ldt:~4,2%%ldt:~6,2%

REM get last update
FOR /F "tokens=* delims=" %%x in (last_update.txt) DO (set OLD_DATE=%%x)

REM get latest Biomedisa version
curl https://biomedisa.info/media/latest_version.txt --output latest_version.txt
FOR /F "tokens=* delims=" %%v in (latest_version.txt) DO (set LATEST_VERSION=%%v)

REM update and run biomedisa engine
if not %LATEST_VERSION% == %VERSION% (
    call :MsgBox "Download and installation may take some time. Upgrade?" "VBYesNo" "There is a new Biomedisa version available."
    if errorlevel 7 (
        wsl -d %VERSION% bash biomedisa_start.sh %VERSION%
    ) else if errorlevel 6 (
        upgrade_biomedisa.cmd
    )
) else if %CURRENT_DATE% GTR %OLD_DATE% (
    call :MsgBox "Update Biomedisa Engine?" "VBYesNo" "Last Update: %OLD_DATE:~0,4%-%OLD_DATE:~4,2%-%OLD_DATE:~6,2%"
    if errorlevel 7 (
        wsl -d %VERSION% bash biomedisa_start.sh %VERSION%
    ) else if errorlevel 6 (
        wsl --terminate %VERSION%
        wsl -d %VERSION% apt-get -y update
        wsl -d %VERSION% apt-get -y dist-upgrade
        wsl -d %VERSION% apt-get -y autoremove
        wsl --terminate %VERSION%
        wsl -d %VERSION% -u biomedisa git -C /home/biomedisa/git/biomedisa/ pull
        wsl -d %VERSION% rsync -avP --exclude 'last_update.txt' /home/biomedisa/git/biomedisa/biomedisa_windows/ "$PWD/"
        wsl -d %VERSION% service mysql start
        wsl -d %VERSION% -u biomedisa python3 /home/biomedisa/git/biomedisa/manage.py migrate
        echo %CURRENT_DATE% > last_update.txt
        wsl --terminate %VERSION%
        wsl -d %VERSION% bash biomedisa_start.sh %VERSION%
    )
) else (wsl -d %VERSION% bash biomedisa_start.sh %VERSION%)

exit

:MsgBox prompt type title
    setlocal enableextensions
    set "tempFile=%temp%\%~nx0.%random%%random%%random%vbs.tmp"
    >"%tempFile%" echo(WScript.Quit msgBox("%~1",%~2,"%~3") & cscript //nologo //e:vbscript "%tempFile%"
    set "exitCode=%errorlevel%" & del "%tempFile%" >nul 2>nul
    endlocal & exit /b %exitCode%

