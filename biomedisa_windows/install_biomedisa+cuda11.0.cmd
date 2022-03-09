@echo off

REM get latest Biomedisa version
curl https://biomedisa.org/media/latest_version_p.txt --output latest_version.txt
FOR /F "tokens=* delims=" %%v in (latest_version.txt) DO (set VERSION=%%v)

REM test Biomedisa installation
wsl -u biomedisa -d %VERSION% touch test_installation.txt
if not exist test_installation.txt (

echo Installing %VERSION%

REM download biomedisa engine
echo Downloading Biomedisa...
curl https://biomedisa.org/media/%VERSION%.tar -C - --output %VERSION%.tar

REM make application directory
mkdir "%USERPROFILE%\AppData\%VERSION%"

REM install biomedisa
echo Biomedisa is being installed. This may take a few minutes. Please wait!
wsl --import %VERSION% "%USERPROFILE%\AppData\%VERSION%" ./%VERSION%.tar

REM verify installation
wsl -u biomedisa -d %VERSION% touch installation_successful.txt
if exist installation_successful.txt (

REM copy biomedisa files
copy biomedisa.ico "%USERPROFILE%\AppData\%VERSION%\biomedisa.ico"
copy biomedisa_start.cmd "%USERPROFILE%\AppData\%VERSION%\biomedisa_start.cmd"
copy biomedisa_start.sh "%USERPROFILE%\AppData\%VERSION%\biomedisa_start.sh"
copy last_update.txt "%USERPROFILE%\AppData\%VERSION%\last_update.txt"
copy biomedisa_interpolation.cmd "%USERPROFILE%\AppData\%VERSION%\biomedisa_interpolation.cmd"
copy biomedisa_interpolation.sh "%USERPROFILE%\AppData\%VERSION%\biomedisa_interpolation.sh"
copy upgrade_biomedisa.cmd "%USERPROFILE%\AppData\%VERSION%\upgrade_biomedisa.cmd"
copy latest_version.txt "%USERPROFILE%\AppData\%VERSION%\version.txt"

REM create shortcut on Desktop
powershell "$s=(New-Object -COM WScript.Shell).CreateShortcut('%userprofile%\Desktop\Biomedisa.lnk');$s.WorkingDirectory='%userprofile%\AppData\%VERSION%';$s.IconLocation='%userprofile%\AppData\%VERSION%\biomedisa.ico';$s.TargetPath='%userprofile%\AppData\%VERSION%\biomedisa_start.cmd';$s.Save()"

REM remove installation files
del %VERSION%.tar
del installation_successful.txt

echo Installation completed.
PAUSE
)

) else (
del test_installation.txt
echo Latest Biomedisa version has already been installed.
PAUSE
)
