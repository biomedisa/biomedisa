# Windows 10/11 (version 21H2 or higher)

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
Open command prompt as **administrator** and run:
```
wsl --install -d Ubuntu-22.04
```
Reboot Windows to complete the installation.

#### Start and update Ubuntu
Open command prompt and start WSL:
```
wsl
```
Update Ubuntu:
```
sudo apt-get update
sudo apt-get dist-upgrade
sudo apt-get autoremove
```

#### Install Biomedisa
Follow the Biomedisa installation instructions for [Ubuntu](https://github.com/biomedisa/biomedisa/#installation-command-line-based).

# Run Biomedisa on WSL
1. **Start WSL**
```
wsl
```
2. **Activate Biomedisa Environment**
```
source biomedisa_env/bin/activate
```
3. **Set Windows Username (Optional)**  
Define your Windows username as a variable to simplify paths:
```
windows_username=$(cmd.exe /c echo %USERNAME% | tr -d '\r')
```
4. **Run Biomedisa Interpolation**  
Use the Windows user directory (typically `/mnt/c/Users`) to specify file paths:
```
python3 -m biomedisa.interpolation \
    /mnt/c/Users/$windows_username/Downloads/tumor.tif \
    /mnt/c/Users/$windows_username/Downloads/labels.tumor.nrrd
```
5. **Skip Environment Activation (Direct Execution)**  
If you prefer not to activate the environment:
```
biomedisa_env/bin/python3 -m biomedisa.interpolation \
    /mnt/c/Users/$windows_username/Downloads/tumor.tif \
    /mnt/c/Users/$windows_username/Downloads/labels.tumor.nrrd
```
6. **Run Directly from Windows Command Prompt**  
To execute without starting WSL manually, use:
```
wsl -e bash -c "export CUDA_HOME=/usr/local/cuda-12.6 && export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 && export PATH=${CUDA_HOME}/bin:${PATH} && windows_username=$(cmd.exe /c echo %USERNAME% | tr -d '\r') && /home/$USER/biomedisa_env/bin/python3 -m biomedisa.interpolation /mnt/c/Users/$windows_username/Downloads/tumor.tif /mnt/c/Users/$windows_username/Downloads/labels.tumor.nrrd"
```
7. **Run Deep Learning Module from Windows Command Prompt**  
No need to set CUDA environment variables:
```
wsl -e bash -c "windows_username=$(cmd.exe /c echo %USERNAME% | tr -d '\r') && /home/$USER/biomedisa_env/bin/python3.10 -m biomedisa.deeplearning /mnt/c/Users/$windows_username/Downloads/mouse_molar_tooth.tif /mnt/c/Users/$windows_username/Downloads/teeth.h5"
```

