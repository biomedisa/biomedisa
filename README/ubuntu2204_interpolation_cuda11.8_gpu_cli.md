# Ubuntu 22.04 LTS + Smart Interpolation + CUDA 11.8 + GPU (command-line-only)

- [Install Python and pip](#install-python-and-pip)
- [Install software dependencies](#install-software-dependencies)
- [Install pip packages](#install-pip-packages)
- [Clone Biomedisa](#clone-biomedisa)
- [Install CUDA 11.8](#install-cuda-11.8)
- [Biomedisa example](#biomedisa-example)

#### Install Python and pip
```
sudo apt-get install python3 python3-dev python3-pip
```

#### Install software dependencies
```
sudo apt-get install libsm6 libxrender-dev \
    libboost-python-dev build-essential libssl-dev cmake \
    openmpi-bin openmpi-doc libopenmpi-dev git libgl1
```

#### Install pip packages
You may only use `pip3 install --upgrade <package>` and add `export PATH=/home/$USER/.local/bin:${PATH}` to `~/.bashrc` if you only install Biomedisa for your user and do not have sudo rights, for example like on a supercomputer
```
sudo -H pip3 install --upgrade pip setuptools testresources scikit-build
sudo -H pip3 install --upgrade numpy scipy h5py colorama numpy-stl \
    numba imagecodecs tifffile scikit-image opencv-python \
    Pillow nibabel medpy SimpleITK mpi4py itk vtk matplotlib
```

#### Clone Biomedisa
```
mkdir ~/git
cd ~/git
git clone https://github.com/biomedisa/biomedisa.git
```

#### Install CUDA 11.8
```
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"

# If W: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg)
sudo apt-key export 3BF863CC | sudo gpg --dearmour -o /etc/apt/trusted.gpg.d/cuda-tools.gpg

# Install CUDA
sudo apt-get update
sudo apt-get install --no-install-recommends cuda-11-8

# Reboot. Check that GPUs are visible using the command
nvidia-smi

# Add the following lines to `~/.bashrc` (e.g. nano ~/.bashrc)
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

# Reload .bashrc and verify that CUDA is installed properly
source ~/.bashrc
nvcc --version

# Install PyCUDA or PyOpenCL
sudo -H "PATH=/usr/local/cuda-11.8/bin:${PATH}" pip3 install --upgrade pycuda

# Verify that PyCUDA is working properly
python3 ~/git/biomedisa/biomedisa_features/pycuda_test.py
```

#### Biomedisa example
Download test files from [Gallery](https://biomedisa.de/gallery/).
```
python3 git/biomedisa/demo/biomedisa_interpolation.py Downloads/tumor.tif Downloads/labels.tumor.tif
```

