# Biomedisa

- [Overview](#overview)
- [Hardware requirements](#hardware-requirements)
- [Software requirements](#software-requirements)
- [Operating systems](#operating-systems)
- [Fast installation of semi-automatic segmentation feature](#fast-installation-of-semi-automatic-segmentation-feature)
- [Run examples](#run-examples)
- [Fast installation of Deep Learning feature](#fast-installation-of-deep-learning-feature)
- [Run example](#run-example)
- [Full installation of Biomedisa online platform](#full-installation-of-biomedisa-online-platform)
- [Update Biomedisa](#update-biomedisa)
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
+ [NVIDIA GPU drivers](https://www.nvidia.com/drivers) - CUDA 11.0 requires 455.x or higher
+ [CUDA Toolkit 11.0](https://developer.nvidia.com/cuda-toolkit-archive)
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
sudo -H pip3 install --upgrade numpy scipy h5py colorama numpy-stl \
    numba imagecodecs-lite tifffile scikit-image opencv-python \
    Pillow SimpleParse nibabel medpy SimpleITK mpi4py itk vtk
```

#### Download or clone Biomedisa
```
sudo apt-get install git
mkdir ~/git
cd ~/git
git clone https://github.com/biomedisa/biomedisa
```

#### Install NVIDIA driver

```
# Install NVIDIA driver >=455
sudo apt-get install nvidia-driver-455

# Reboot the system
sudo reboot

# Verify that NVIDIA driver can be loaded properly
nvidia-smi
```

#### Install CUDA 11.0

```
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get install --no-install-recommends cuda-11-0

# Reboot. Check that GPUs are visible using the command
nvidia-smi

# Add the following lines to your .bashrc (e.g. nano ~/.bashrc)
export CUDA_HOME=/usr/local/cuda-11.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

# Reload .bashrc and verify that CUDA is installed properly
source ~/.bashrc
nvcc --version

# Install PyCUDA
sudo -H PATH=/usr/local/cuda-11.0/bin:${PATH} pip3 install --upgrade pycuda

# Verify that PyCUDA is working properly
python3 ~/git/biomedisa/biomedisa_features/pycuda_test.py
```

## Windows 10

#### Install Microsoft Visual Studio 2017
Download and install [MS Visual Studio](https://visualstudio.microsoft.com/de/thank-you-downloading-visual-studio/?sku=Community&rel=15&rr=https%3A%2F%2Fwww.wintotal.de%2Fdownload%2Fmicrosoft-visual-studio-community-2017%2F).
```
Select "Desktop development with C++"
Install
Restart Windows
```

#### Set Path Variables
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
pip3 install --upgrade nibabel medpy SimpleITK simpleparse itk vtk numpy-stl
```

#### Install NVIDIA Driver
Download and install [NVIDIA](https://www.nvidia.com/Download/Find.aspx?lang=en-us).  
Choose *Windows Driver Type:* Standard  
Choose *Recommended/Beta:* Studio Driver

#### Install CUDA Toolkit 11.0
Download and install [CUDA Toolkit 11.0](https://developer.nvidia.com/cuda-downloads).

#### Disable TDR in Nsight Monitor
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

#### Run small example
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

#### Run further examples
Download the examples from the [gallery](https://biomedisa.org/gallery/) or directly as follows:
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

Obtain uncertainty and smoothing as optional results
```
mpiexec -n 4 python3 biomedisa_interpolation.py NMB_F2875.tif labels.NMB_F2875.tif -uq -s 100
```

Use pre-segmentation in different planes (not exclusively xy-plane)
```
mpiexec -n 4 python3 'path_to_image' 'path_to_labels' -allx
```
# Fast installation of Deep Learning feature

## Ubuntu 18.04.5
If you have not already done so, follow the installation instructions [Fast installation of semi-automatic segmentation feature](#fast-installation-of-semi-automatic-segmentation-feature) (Ubuntu 18.04.5). Then install TensorFlow.

#### Install TensorFlow
```
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt install ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt-get update

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    libcudnn8=8.0.4.30-1+cuda11.0  \
    libcudnn8-dev=8.0.4.30-1+cuda11.0

# Install TensorRT. Requires that libcudnn8 is installed above.
sudo apt-get install -y --no-install-recommends libnvinfer7=7.1.3-1+cuda11.0 \
    libnvinfer-dev=7.1.3-1+cuda11.0 \
    libnvinfer-plugin7=7.1.3-1+cuda11.0

# Install TensorFlow
sudo -H pip3 install tensorflow-gpu==2.4.1
```
## Windows 10
If you have not already done so, follow the installation instructions [Fast installation of semi-automatic segmentation feature](#fast-installation-of-semi-automatic-segmentation-feature) (Windows 10). Then install TensorFlow.

#### Install cuDNN
Download [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) (free NVIDIA account required).  
Extract the ZIP folder.

#### Set Path Variables
Open Windows Search  
Type `View advanced system settings`  
Click `Environment Variables...`  
Add the following value to the **System variable** `Path`
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\extras\CUPTI\lib64
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include
C:\Users\USERNAME\cuda\bin      (The path where you extracted cuDNN)
```
#### Install TensorFlow
```
pip3 install tensorflow-gpu==2.4.1
```

# Run example
Change to the demo directory.
```
cd git/biomedisa/demo
```

Download the Deep Learning example `human heart` from the [gallery](https://biomedisa.org/gallery/) or directly as follows:
```
wget https://biomedisa.org/media/training_hearts.tar
wget https://biomedisa.org/media/training_hearts_labels.tar
wget https://biomedisa.org/media/testing_axial_crop_pat13.nii.gz
```

Extract the data. This creates a `heart` directory containing the image data and a `label` directory containing the label data.
```
tar -xf training_hearts.tar
tar -xf training_hearts_labels.tar
```

Train a neural network with 200 epochs and batch size (-bs) of 24. The result will be saved as `heart.h5`. If you have only a single GPU, reduce batch size to 6.
```
# Ubuntu
python3 biomedisa_deeplearning.py heart label -train -epochs 200 -bs 24

# Windows
python biomedisa_deeplearning.py heart label -train -epochs 200 -bs 24
```

Alternatively, you can download the trained network from the [gallery](https://biomedisa.org/gallery/) or directly with the command
```
wget https://biomedisa.org/media/heart.h5
```

Use the trained network to predict the result of the test image. The result will be saved as `final.testing_axial_crop_pat13.tif`.
```
# Ubuntu
python3 biomedisa_deeplearning.py testing_axial_crop_pat13.nii.gz heart.h5 -predict -bs 6

# Windows
python biomedisa_deeplearning.py testing_axial_crop_pat13.nii.gz heart.h5 -predict -bs 6
```

# Full installation of Biomedisa online platform

### Ubuntu 18.04.5

Please follow the [installation instructions](https://github.com/biomedisa/biomedisa/blob/master/README/INSTALL_UBUNTU_18.04.5.md).

### Windows 10

Please follow the [installation instructions](https://github.com/biomedisa/biomedisa/blob/master/README/INSTALL_WINDOWS_10.md).

# Update Biomedisa
If you've used `git clone` change to the Biomedisa directory and make a pull request
```
cd git/biomedisa
git pull
```

When you have fully installed Biomedisa (including the MySQL database), update the database 
```
python3 manage.py migrate
```

If you installed an [Apache Server](https://github.com/biomedisa/biomedisa/blob/master/README/INSTALL_APACHE_SERVER.md), restart the server
```
sudo service apache2 restart
```

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

