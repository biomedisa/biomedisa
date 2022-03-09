# Windows 10 (version 21H2 only) and Windows 11
With this you set up an already configured version of Biomedisa in a virtual machine with WSL (~23 GB). This only works on Windows 10 (version 21h2) and Windows 11. If necessary, you can update your system under the Windows settings "Update & Security" or with the [Windows Update Assistant](https://support.microsoft.com/en-us/topic/windows-10-update-assistant-3550dfb2-a015-7765-12ea-fba2ac36fb3f). The Biomedisa installation will be located in `C:\Users\username\AppData\Biomedisa-2x.xx.x`.

- [Install NVIDIA driver](#install-nvidia-driver)
- [Enable "Virtualization" in the BIOS](#enable-virtualization-in-the-bios)
- [Install WSL 2 with administrative privileges](#install-wsl-2-with-administrative-privileges)
- [Download Biomedisa installer](#download-biomedisa-installer)
- [Run installation script](#run-installation-script)
- [Start Biomedisa](#start-biomedisa)
- [Stop Biomedisa](#stop-biomedisa)
- [Uninstallation](#uninstallation)

#### Install NVIDIA driver
Download and install [NVIDIA](https://www.nvidia.com/Download/Find.aspx?lang=en-us) driver.

#### Enable "Virtualization" in the BIOS
At Intel it is typically called "Intel Virtualization Technology" and can be found under "CPU configuration". You may arrive at this menu by clicking on “Advanced” or “Advanced Mode”. Depending upon your PC, look for any of these or similar names such as Hyber-V, Vanderpool, SVM, AMD-V, Intel Virtualization Technology or VT-X.

#### Install WSL 2 with administrative privileges and reboot
```
wsl --install
```

#### Download Biomedisa installer
[Biomedisa + Windows](https://biomedisa.org/media/biomedisa_windows.zip)

#### Run installation script
For NVIDIA Pascal and Volta GPUs
```
install_biomedisa+cuda11.0.cmd
```
For NVIDIA Ampere GPUs
```
install_biomedisa+cuda11.4.cmd
```

#### Start Biomedisa
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
Remove specific Biomedisa version.
```
wsl --unregister Biomedisa-2x.xx.x
```
