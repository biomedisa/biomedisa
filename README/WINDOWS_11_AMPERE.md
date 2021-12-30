# Windows 10 (version 21H2 only) and Windows 11
With this you set up an already configured version of Biomedisa in a virtual machine with WSL (~23 GB). This only works on Windows 10 (version 21h2) and Windows 11. The installation will be located in `C:\Users\username\AppData\Biomedisa-21.12.1`.

- [Install NVIDIA driver](#install-nvidia-driver)
- [Install WSL2 with administrative privileges](#install-wsl2-with-administrative-privileges)
- [Reboot and activate "Virtualization" in the BIOS](#reboot-and-activate-virtualization-in-the-bios)
- [Download and extract Biomedisa](#download-and-extract-biomedisa)
- [Run installation script](#run-install-script)
- [Start Biomedisa using the shortcut](#start-biomedisa-using-the-shortcut)
- [Remove installation files](#remove-installation-files)

#### Install NVIDIA driver
Download and install [NVIDIA](https://www.nvidia.com/Download/Find.aspx?lang=en-us).

#### Install WSL2 with administrative privileges
```
wsl --install
```

#### Reboot and activate "Virtualization" in the BIOS
At Intel it is typically called "Intel Virtualization Technology".

#### Download and extract installation files
Download and extract [Biomedisa](https://biomedisa.org/media/Biomedisa-21.12.1.zip).

#### Run installation script
```
install.cmd
```

#### Start Biomedisa
Login as superuser "biomedisa" with password "biomedisa".

#### Remove installation files

