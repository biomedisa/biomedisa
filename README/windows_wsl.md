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
For simplicity, we use relative paths and assume your Biomedisa environment is located in your Windows user directory (`C:\Users\WINDOWS_USERNAME`), which corresponds to `/mnt/c/Users/WINDOWS_USERNAME` in WSL:
1. **Start WSL**
```
wsl
```
2. **Activate Biomedisa Environment**
```
source biomedisa_env/bin/activate
```
3. **Run Biomedisa Interpolation**  
```
python3 -m biomedisa.interpolation Downloads/tumor.tif Downloads/labels.tumor.nrrd
```
4. **Skip Environment Activation (Direct Execution)**  
If you prefer not to activate the environment:
```
biomedisa_env/bin/python3 -m biomedisa.interpolation Downloads/tumor.tif Downloads/labels.tumor.nrrd
```
5. **Run Directly from Windows Command Prompt**  
To execute without starting WSL manually, use:
```
wsl -e bash -c "export CUDA_HOME=/usr/local/cuda-12.6 && export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 && export PATH=${CUDA_HOME}/bin:${PATH} && biomedisa_env/bin/python3 -m biomedisa.interpolation Downloads/tumor.tif Downloads/labels.tumor.nrrd"
```
6. **Run Deep Learning Module from Windows Command Prompt**  
No need to set CUDA environment variables:
```
wsl -e bash -c "biomedisa_env/bin/python3 -m biomedisa.deeplearning Downloads/mouse_molar_tooth.tif Downloads/teeth.h5"
```

