@echo off

REM get latest Biomedisa version
curl https://biomedisa.org/media/latest_version.txt --output latest_version.txt
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

REM download biomedisa files
curl https://raw.githubusercontent.com/biomedisa/biomedisa/master/biomedisa_installer/biomedisa.ico --output "%USERPROFILE%\AppData\%VERSION%\biomedisa.ico"
curl https://raw.githubusercontent.com/biomedisa/biomedisa/master/biomedisa_installer/biomedisa_start.cmd --output "%USERPROFILE%\AppData\%VERSION%\biomedisa_start.cmd"
curl https://raw.githubusercontent.com/biomedisa/biomedisa/master/biomedisa_installer/biomedisa_start.sh --output "%USERPROFILE%\AppData\%VERSION%\biomedisa_start.sh"
curl https://raw.githubusercontent.com/biomedisa/biomedisa/master/biomedisa_installer/last_update.txt --output "%USERPROFILE%\AppData\%VERSION%\last_update.txt"
curl https://raw.githubusercontent.com/biomedisa/biomedisa/master/biomedisa_installer/biomedisa_interpolation.cmd --output "%USERPROFILE%\AppData\%VERSION%\biomedisa_interpolation.cmd"
curl https://raw.githubusercontent.com/biomedisa/biomedisa/master/biomedisa_installer/biomedisa_interpolation.sh --output "%USERPROFILE%\AppData\%VERSION%\biomedisa_interpolation.sh"
curl https://raw.githubusercontent.com/biomedisa/biomedisa/master/biomedisa_installer/upgrade_biomedisa.cmd --output "%USERPROFILE%\AppData\%VERSION%\upgrade_biomedisa.cmd"

REM copy biomedisa files
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
