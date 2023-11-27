@echo off

REM get current Biomedisa version
FOR /F "tokens=* delims=" %%v in (version.txt) DO (set OLD_VERSION=%%v)

REM get latest Biomedisa version
curl https://biomedisa.info/media/latest_version.txt --output latest_version.txt
FOR /F "tokens=* delims=" %%v in (latest_version.txt) DO (set VERSION=%%v)

REM get current date
for /f "tokens=1,2 delims==" %%i in ('wmic os get LocalDateTime /VALUE') do (if %%i EQU LocalDateTime set ldt=%%j)
set CURRENT_DATE=%ldt:~0,4%%ldt:~4,2%%ldt:~6,2%

REM verify old installation
wsl -u biomedisa -d %VERSION% touch installation_exists.txt

REM check if new version is available
if %VERSION% == %OLD_VERSION% (
echo Biomedisa is already the latest version.
if exist installation_exists.txt (
    del installation_exists.txt
    )
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
curl https://biomedisa.info/media/%VERSION%.tar -C - --output %VERSION%.tar

REM make application directory
mkdir "%USERPROFILE%\AppData\%VERSION%"

REM install biomedisa
echo Biomedisa is being installed. This may take a few minutes. Please wait!
wsl --import %VERSION% "%USERPROFILE%\AppData\%VERSION%" ./%VERSION%.tar
timeout /t 5 /nobreak >nul

REM verify new installation
wsl -u biomedisa -d %VERSION% touch upgrade_successful.txt
if exist upgrade_successful.txt (

REM update old engine
wsl -d %OLD_VERSION% -u biomedisa git -C /home/biomedisa/git/biomedisa/ pull
wsl -d %OLD_VERSION% rsync -avP --exclude 'last_update.txt' /home/biomedisa/git/biomedisa/biomedisa_windows/ "$PWD/"
wsl -d %OLD_VERSION% service mysql start
wsl -d %OLD_VERSION% -u biomedisa python3 /home/biomedisa/git/biomedisa/manage.py migrate
wsl -d %OLD_VERSION% service mysql stop

REM update new engine
wsl --terminate %VERSION%
wsl -d %VERSION% apt-get -y update
wsl -d %VERSION% apt-get -y dist-upgrade
wsl -d %VERSION% apt-get -y autoremove
wsl --terminate %VERSION%
wsl -d %VERSION% -u biomedisa git -C /home/biomedisa/git/biomedisa/ pull
wsl -d %VERSION% service mysql start
wsl -d %VERSION% -u biomedisa python3 /home/biomedisa/git/biomedisa/manage.py migrate
echo %CURRENT_DATE% > last_update.txt
wsl --terminate %VERSION%

REM synchronize data
wsl -u biomedisa -d %OLD_VERSION% mkdir /mnt/wsl/share
wsl -u biomedisa -d %OLD_VERSION% rsync -avP /home/biomedisa/git/biomedisa/log /mnt/wsl/share
wsl -u biomedisa -d %OLD_VERSION% rsync -avP /home/biomedisa/git/biomedisa/private_storage /mnt/wsl/share

wsl -u biomedisa -d %VERSION% rsync -avP /mnt/wsl/share/log/ /home/biomedisa/git/biomedisa/log/
wsl -u biomedisa -d %VERSION% rsync -avP /mnt/wsl/share/private_storage/ /home/biomedisa/git/biomedisa/private_storage/

REM synchronize database
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

