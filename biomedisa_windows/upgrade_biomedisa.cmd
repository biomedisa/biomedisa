@echo off

REM get current Biomedisa version
FOR /F "tokens=* delims=" %%v in (version.txt) DO (set OLD_VERSION=%%v)

REM get latest Biomedisa version
if %OLD_VERSION:~-1% == p (
    curl https://biomedisa.org/media/latest_version_p.txt --output latest_version.txt
) else (
    curl https://biomedisa.org/media/latest_version.txt --output latest_version.txt
)
FOR /F "tokens=* delims=" %%v in (latest_version.txt) DO (set VERSION=%%v)

REM verify old installation
wsl -u biomedisa -d %VERSION% touch installation_exists.txt

REM check if new version is available
if %VERSION% == %OLD_VERSION% (
echo Biomedisa is already the latest version.
PAUSE

) else if exist installation_exists.txt (
echo Biomedisa is already the latest version.
del installation_exists.txt
PAUSE

) else (

REM upgrade Biomedisa
echo Installing %VERSION%

REM download biomedisa engine
echo Downloading Biomedisa...
curl https://biomedisa.org/media/%VERSION%.tar -C - --output %VERSION%.tar

REM make application directory
mkdir "%USERPROFILE%\AppData\%VERSION%"

REM install biomedisa
echo Biomedisa is being installed. This may take a few minutes. Please wait!
wsl --import %VERSION% "%USERPROFILE%\AppData\%VERSION%" ./%VERSION%.tar

REM verify new installation
wsl -u biomedisa -d %VERSION% touch upgrade_successful.txt
if exist upgrade_successful.txt (

REM synchronize data
wsl -u biomedisa -d %OLD_VERSION% mkdir /mnt/wsl/share
wsl -u biomedisa -d %OLD_VERSION% rsync -avP /home/biomedisa/git/biomedisa/log /mnt/wsl/share
wsl -u biomedisa -d %OLD_VERSION% rsync -avP /home/biomedisa/git/biomedisa/private_storage /mnt/wsl/share

wsl -u biomedisa -d %VERSION% rsync -avP /mnt/wsl/share/log/ /home/biomedisa/git/biomedisa/log/
wsl -u biomedisa -d %VERSION% rsync -avP /mnt/wsl/share/private_storage/ /home/biomedisa/git/biomedisa/private_storage/

wsl -d %OLD_VERSION% service mysql start
wsl -u biomedisa -d %OLD_VERSION% mysqldump -u root -pbiomedisa --opt biomedisa_database > biomedisa_database.sql
wsl -d %OLD_VERSION% service mysql stop

wsl -d %VERSION% service mysql start
wsl -u biomedisa -d %VERSION% mysql -u root -pbiomedisa biomedisa_database < biomedisa_database.sql
wsl -d %VERSION% service mysql stop

REM copy biomedisa files
copy biomedisa.ico "%USERPROFILE%\AppData\%VERSION%\biomedisa.ico"
copy biomedisa_start.cmd "%USERPROFILE%\AppData\%VERSION%\biomedisa_start.cmd"
copy biomedisa_start.sh "%USERPROFILE%\AppData\%VERSION%\biomedisa_start.sh"
copy last_update.txt "%USERPROFILE%\AppData\%VERSION%\last_update.txt"
copy latest_version.txt "%USERPROFILE%\AppData\%VERSION%\version.txt"
copy biomedisa_interpolation.cmd "%USERPROFILE%\AppData\%VERSION%\biomedisa_interpolation.cmd"
copy biomedisa_interpolation.sh "%USERPROFILE%\AppData\%VERSION%\biomedisa_interpolation.sh"
copy upgrade_biomedisa.cmd "%USERPROFILE%\AppData\%VERSION%\upgrade_biomedisa.cmd"

REM create shortcut on Desktop
powershell "$s=(New-Object -COM WScript.Shell).CreateShortcut('%userprofile%\Desktop\Biomedisa.lnk');$s.WorkingDirectory='%userprofile%\AppData\%VERSION%';$s.IconLocation='%userprofile%\AppData\%VERSION%\biomedisa.ico';$s.TargetPath='%userprofile%\AppData\%VERSION%\biomedisa_start.cmd';$s.Save()"

REM old biomedisa version
wsl --unregister %OLD_VERSION%

REM remove installation files
del %VERSION%.tar
del upgrade_successful.txt

echo Upgrade completed.
PAUSE
)

)