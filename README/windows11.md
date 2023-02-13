# Windows 10 (version 21H2 or higher) and Windows 11
With this you set up an already configured version of Biomedisa in a virtual machine with WSL (~30 GB). This only works on Windows 10 (version 21H2 or higher) and Windows 11. If necessary, you can update your system under the Windows settings "Update & Security" or with the [Windows Update Assistant](https://support.microsoft.com/en-us/topic/windows-10-update-assistant-3550dfb2-a015-7765-12ea-fba2ac36fb3f). The Biomedisa installation will be located in `C:\Users\username\AppData\Biomedisa-2x.xx.x`.

- [Install NVIDIA driver](#install-nvidia-driver)
- [Enable "Virtualization" in the BIOS](#enable-virtualization-in-the-bios)
- [Install WSL 2 with administrative privileges](#install-wsl-2-with-administrative-privileges)
- [Reboot Windows](#reboot-windows)
- [Download Biomedisa installer](#download-biomedisa-installer)
- [Run installation script](#run-installation-script)
- [Start Biomedisa](#start-biomedisa)
- [Stop Biomedisa](#stop-biomedisa)
- [Uninstallation](#uninstallation)

#### Install NVIDIA driver (only required for NVIDIA GPUs)
Download and install [NVIDIA](https://www.nvidia.com/Download/Find.aspx?lang=en-us) driver.

#### Enable "Virtualization" in the BIOS
At Intel it is typically called "Intel Virtualization Technology" and can be found under "CPU configuration". You may arrive at this menu by clicking on “Advanced” or “Advanced Mode”. Depending upon your PC, look for any of these or similar names such as Hyber-V, Vanderpool, SVM, AMD-V, Intel Virtualization Technology or VT-X.

#### Install WSL 2 with administrative privileges
```
wsl --install
```
#### Reboot Windows to complete the installation

#### Remove the initial Ubuntu installation to save space (optional)
```
wsl --unregister Ubuntu
```
#### Download Biomedisa installer
Download and extract [Biomedisa](https://biomedisa.org/media/biomedisa_windows.zip).

#### Run installation script
```
install_biomedisa.cmd
```
If the installation is interrupted, please run the script again.

#### Start Biomedisa
Use the desktop icon or open localhost in your browser.
```
http://localhost
```
Login as superuser "biomedisa" with password "biomedisa".

#### Stop Biomedisa
```
wsl --shutdown
```
#### Uninstallation
Find your Biomedisa version.
```
wsl -l -v
```
Remove a specific Biomedisa version.
```
wsl --unregister Biomedisa-2x.xx.x
```

