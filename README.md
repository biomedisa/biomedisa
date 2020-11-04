# Biomedisa

- [Overview](#overview)
- [Hardware requirements](#hardware-requirements)
- [Software requirements](#software-requirements)
- [Operating systems](#operating-systems)
- [Fast installation of semi-automatic segmentation feature](#fast-installation-of-semi-automatic-segmentation-feature)
- [Run examples](#run-examples)
- [Full installation of Biomedisa online platform](#Full-installation-of-biomedisa-online-platform)
- [Releases](#releases)
- [Authors](#authors)
- [FAQ](#faq)
- [Citation](#citation)
- [License](#license)

# Overview
Biomedisa (https://biomedisa.org) is a free and easy-to-use open-source online platform for segmenting large volumetric images, e.g. CT and MRI scans, at Heidelberg University and the Heidelberg Institute for Theoretical Studies (HITS). The segmentation is based on a smart interpolation of sparsely pre-segmented slices taking into account the complete underlying image data. It can be used in addition to segmentation tools like Amira, ImageJ/Fiji and MITK. Biomedisa finds its root in the projects ASTOR and NOVA funded by the Federal Ministry of Education and Research (BMBF). If you are using Biomedisa for your research please cite: Lösel, P.D. et al. [Introducing Biomedisa as an open-source online platform for biomedical image segmentation.](https://www.nature.com/articles/s41467-020-19303-w) *Nat. Commun.* **11**, 5577 (2020).

# Hardware requirements
+ At least one [NVIDIA](https://www.nvidia.com/) Graphics Procissing Unit (GPU) with compute capability 3.0 or higher.
+ 32 GB RAM or more (strongly depends on the size of the processed images).

# Software requirements
+ [NVIDIA GPU drivers](https://www.nvidia.com/drivers) - CUDA 10.1 requires 418.x or higher
+ [CUDA Toolkit 10.1](https://developer.nvidia.com/cuda-toolkit-archive)
+ [Open MPI 2.1.1](https://www.open-mpi.org/)

# Operating systems
This package is supported for **Linux** and **Windows** and has been tested on the following systems:
+ Linux: Ubuntu 18.04.5
+ Windows: Windows 10

# Fast installation of semi-automatic segmentation feature 

## Ubuntu 18.04.5
The installation takes approximatly 15 minutes (SSD).

#### Install Python and pip
```
sudo apt-get install python3 python3-dev python3-pip
```

#### Install software dependencies
```
sudo apt-get install libsm6 libxrender-dev cmake \
    libboost-python-dev build-essential screen libssl-dev \
    openmpi-bin openmpi-doc libopenmpi-dev
```

#### Install pip packages
```
sudo -H pip3 install --upgrade pip setuptools scikit-build 
sudo -H pip3 install --upgrade numpy scipy h5py colorama \
    numba imagecodecs-lite tifffile scikit-image opencv-python \
    Pillow SimpleParse nibabel medpy SimpleITK mpi4py
```

#### Download or clone Biomedisa
```
sudo apt-get install git
mkdir ~/git
cd ~/git
git clone https://github.com/biomedisa/biomedisa
```

#### Setting up CUDA environment

```
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update

# Install NVIDIA driver
sudo apt-get install --no-install-recommends nvidia-driver-450

# Reboot the system
sudo reboot

# Verify that NVIDIA driver can be loaded properly
nvidia-smi

# Install CUDA 10.1
sudo apt-get install --no-install-recommends cuda-10-1

# Add the following lines to your .bashrc (e.g. nano ~/.bashrc)
export CUDA_HOME=/usr/local/cuda-10.1
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

# Reload .bashrc and verify that CUDA is installed properly
source ~/.bashrc
nvcc --version

# Install PyCUDA
sudo -H PATH=/usr/local/cuda-10.1/bin:${PATH} pip3 install --upgrade pycuda

# Verify that PyCUDA is working properly
python3 ~/git/biomedisa/biomedisa_features/pycuda_test.py
```

## Windows 10

### Install Microsoft Visual Studio 2017
Download and install [MS Visual Studio](https://visualstudio.microsoft.com/de/thank-you-downloading-visual-studio/?sku=Community&rel=15&rr=https%3A%2F%2Fwww.wintotal.de%2Fdownload%2Fmicrosoft-visual-studio-community-2017%2F).
```
Select "Desktop development with C++"
Install
Restart Windows
```

### Set Path Variables
Open Windows Search  
Type `View advanced system settings`  
Click `Environment Variables...`  
Add the following value to the **System variable** `Path`
```
C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64
```

#### Install Python and pip
Download and install [Python](https://www.python.org/downloads/windows/).
```
Choose "Latest Python 3 Release"
Scroll to "Files"
Choose "Windows x86-64 executable installer"
Select "Add Python 3.X to PATH"
Install
Disable path length limit
Close
```

#### Install pip packages
Open Command Prompt (e.g. Windows Search `Command Prompt`).
```
pip3 install --upgrade pip pypiwin32 setuptools wheel numpy scipy h5py colorama numba
pip3 install --upgrade imagecodecs-lite tifffile scikit-image opencv-python Pillow
pip3 install --upgrade nibabel medpy SimpleITK simpleparse
```

#### Install NVIDIA Driver
Download and install [NVIDIA](https://www.nvidia.com/Download/Find.aspx?lang=en-us).  
Choose *Windows Driver Type:* Standard  
Choose *Recommended/Beta:* Studio Driver

#### Install CUDA Toolkit 11.0
Download and install [CUDA Toolkit 11.0](https://developer.nvidia.com/cuda-downloads).

### Disable TDR in Nsight Monitor
```
Windows Search "Nsight Monitor" (run as administrator)
Click the "NVIDIA Nsight" symbol in the right corner of your menu bar
Click Nsight Monitor options
Disable "WDDM TDR Enabled"
Reboot your system
```

#### Install Microsoft MPI
Download and install [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467).
```
Select "msmpisetup.exe" to download
Install
```

#### Install PyCUDA and mpi4py
Open Command Prompt (e.g. Windows Search `Command Prompt`).
```
pip3 install --upgrade pycuda mpi4py
```

#### Download or clone Biomedisa
Download and install [Git](https://github.com/git-for-windows/git/releases/download/v2.28.0.windows.1/Git-2.28.0-64-bit.exe).
```
mkdir git
cd git
git clone https://github.com/biomedisa/biomedisa
```

#### Verify that PyCUDA is working properly
```
python biomedisa/biomedisa_features/pycuda_test.py
```

# Run examples

### Run small examples
Change to the demo directory.
```
cd git/biomedisa/demo
```

Run a simple example (~4 seconds). The result will be saved as `final.tumor.tif`.
```
# Ubuntu
python3 biomedisa_interpolation.py tumor.tif labels.tumor.tif

# Windows
python biomedisa_interpolation.py tumor.tif labels.tumor.tif
```

### Run further examples
Download the examples from https://biomedisa.org/gallery/ or directly as follows:
```
# Trigonopterus
wget --no-check-certificate https://biomedisa.org/download/demo/3 -O trigonopterus.tif
wget --no-check-certificate https://biomedisa.org/download/demo/4 -O labels.trigonopterus_smart.am

# Wasp from amber
wget --no-check-certificate https://biomedisa.org/download/demo/28 -O wasp_from_amber.tif
wget --no-check-certificate https://biomedisa.org/download/demo/29 -O labels.wasp_from_amber.am

# Cockroach
wget --no-check-certificate https://biomedisa.org/download/demo/31 -O cockroach.tif
wget --no-check-certificate https://biomedisa.org/download/demo/32 -O labels.cockroach.am

# Theropod claw
wget --no-check-certificate https://biomedisa.org/download/demo/34 -O theropod_claw.tif
wget --no-check-certificate https://biomedisa.org/download/demo/35 -O labels.theropod_claw.tif

# Mineralized wasp
wget --no-check-certificate https://biomedisa.org/download/demo/17 -O NMB_F2875.tif
wget --no-check-certificate https://biomedisa.org/download/demo/18 -O labels.NMB_F2875.tif

# Bull ant
wget --no-check-certificate https://biomedisa.org/download/demo/37 -O bull_ant.tif
wget --no-check-certificate https://biomedisa.org/download/demo/38 -O labels.bull_ant_head.am
```

Run the segmentation using e.g. 4 GPUs
```
mpiexec -n 4 python3 biomedisa_interpolation.py NMB_F2875.tif labels.NMB_F2875.tif
```

Obtain uncertainty and smooting as optional results
```
mpiexec -n 4 python3 biomedisa_interpolation.py NMB_F2875.tif labels.NMB_F2875.tif -uq -s 100
```

Enable labeling in different planes (not only xy-plane)
```
mpiexec -n 4 python3 'path_to_image' 'path_to_labels' -allx
```

# Full installation of Biomedisa online platform

### Ubuntu 18.04.5

Please follow the [installation instructions](https://github.com/biomedisa/biomedisa/blob/master/README/INSTALL_UBUNTU_18.04.5.md). Alternatively, clone the repository and run the install script `install.sh`.

### Windows 10

Please follow the [installation instructions](https://github.com/biomedisa/biomedisa/blob/master/README/INSTALL_WINDOWS_10.md).

# Releases

For the versions available, see the [tags on this repository](https://github.com/biomedisa/biomedisa/tags). 

# Authors

* **Philipp D. Lösel**

See also the list of [contributors](https://github.com/biomedisa/biomedisa/blob/master/credits.md) who participated in this project.

# FAQ
Frequently asked questions can be found at: https://biomedisa.org/faq/.

# Citation

If you use the package or the online platform, please cite the following paper.

`Lösel, P.D. et al. Introducing Biomedisa as an open-source online platform for biomedical image segmentation. Nat. Commun. 11, 5577 (2020).`

# License

This project is covered under the **EUROPEAN UNION PUBLIC LICENCE v. 1.2 (EUPL)**.

