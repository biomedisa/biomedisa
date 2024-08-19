# Windows 10 (version 21H2 or higher) and Windows 11

- [Install NVIDIA driver](#install-nvidia-driver)
- [Enable "Virtualization" in the BIOS](#enable-virtualization-in-the-bios)
- [Install WSL with administrative privileges](#install-wsl-2-with-administrative-privileges)
- [Start and update Ubuntu](#dstart-and-update-ubuntu)
- [Install Biomedisa](#install-biomedisa)
- [Run Biomedisa](#run-biomedisa)

#### Install NVIDIA driver
Download and install [NVIDIA](https://www.nvidia.com/Download/Find.aspx?lang=en-us) driver.

#### Enable "Virtualization" in the BIOS (for new systems this may not be necessary)
At Intel it is typically called "Intel Virtualization Technology" and can be found under "CPU configuration". You may arrive at this menu by clicking on “Advanced” or “Advanced Mode”. Depending upon your PC, look for any of these or similar names such as Hyber-V, Vanderpool, SVM, AMD-V, Intel Virtualization Technology or VT-X.

#### Install WSL and Ubuntu 22.04 with administrative privileges
Open command prompt with administrative privileges and run:
```
wsl --install -d Ubuntu-22.04
```
Reboot Windows to complete the installation.

#### Start and update Ubuntu
Start WSL:
```
wsl
```
Update Ubuntu:
```
sudo apt-get update
sudo apt-get dist-upgrade
```

#### Install Biomedisa
Follow the Biomedisa installation instructions for [Ubuntu](https://github.com/biomedisa/biomedisa/#installation-command-line-based).


#### Run Biomedisa
Within WSL you will usually find your Windows user directory under `/mnt/c/Users/$USER`, e.g.:
```
python3 -m biomedisa.interpolation /mnt/c/Users/$USER/Downloads/tumor.tif /mnt/c/Users/$USER/Downloads/labels.tumor.tif

```

