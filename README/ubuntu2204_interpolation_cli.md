# Ubuntu 22.04 LTS + Smart Interpolation (command-line-only)

- [Install Python and pip](#install-python-and-pip)
- [Install software dependencies](#install-software-dependencies)
- [Install CUDA Toolkit](#install-cuda-toolkit)
- [Install pip packages](#install-pip-packages)
- [Biomedisa examples](#biomedisa-examples)
- [Install Biomedisa from source (optional)](#install-biomedisa-from-source-optional)

#### Install Python and pip
```
sudo apt-get install python3 python3-dev python3-pip
```

#### Install software dependencies
```
sudo apt-get install libsm6 libxrender-dev unzip \
    libboost-python-dev build-essential libssl-dev cmake \
    openmpi-bin openmpi-doc libopenmpi-dev libgl1 wget
```

#### Install CUDA Toolkit
Download and install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) or via command-line as follows. You may choose any CUDA version compatible with your NVIDIA GPU architecture as outlined in the [NVIDIA Documentation](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html). If you select a version other than 12.6, you will need to adjust the following steps accordingly:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

#### Verifying Driver Installation
Reboot and verify whether the NVIDIA drivers are installed and working properly by running:
```
nvidia-smi
```

#### Installing NVIDIA Drivers Separately
If the NVIDIA drivers are **not** installed, you can install them separately with:
```
sudo apt install nvidia-driver-<version>
```

#### Adapt PATH variables
Add the local pip directory to the PATH variable:
```
echo 'export PATH=${HOME}/.local/bin:${PATH}' >> ~/.bashrc
```
Add the CUDA paths:
```
echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64' >> ~/.bashrc
echo 'export PATH=${CUDA_HOME}/bin:${PATH}' >> ~/.bashrc
```
Reload .bashrc and verify that CUDA is installed properly:
```
source ~/.bashrc
nvcc --version
```

#### Install pip packages
```
python3 -m pip install pip setuptools testresources scikit-build
python3 -m pip install numpy scipy h5py colorama numpy-stl \
    numba imagecodecs tifffile scikit-image opencv-python netCDF4 mrcfile \
    Pillow nibabel medpy SimpleITK mpi4py itk vtk matplotlib biomedisa \
    importlib_metadata
PATH=/usr/local/cuda-12.6/bin:${PATH} python3 -m pip install pycuda
```

#### Verify that PyCUDA is working properly
```
python3 -m biomedisa.features.pycuda_test
```

#### Biomedisa Examples
Download test files from [Gallery](https://biomedisa.info/gallery/) or via command-line:
```
wget -P ~/Downloads/ https://biomedisa.info/media/images/tumor.tif
wget -P ~/Downloads/ https://biomedisa.info/media/images/labels.tumor.nrrd
```
Smart Interpolation:
```
python3 -m biomedisa.interpolation ~/Downloads/tumor.tif ~/Downloads/labels.tumor.nrrd
```

#### Install Biomedisa from source (optional)
To develop Biomedisa or for the latest version install Biomedisa from [source](https://github.com/biomedisa/biomedisa/blob/master/README/installation_from_source.md).

