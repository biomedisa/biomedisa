# Ubuntu 20.04.3 LTS + CUDA 11.0 (command-line-only)

- [Install Python and pip](#install-python-and-pip)
- [Install software dependencies](#install-software-dependencies)
- [Install pip packages](#install-pip-packages)
- [Download or clone Biomedisa](#download-or-clone-biomedisa)
- [Install CUDA 11.0](#install-cuda-11.0)
- [Install TensorFlow](#install-tensorflow)

#### Install Python and pip
```
sudo apt-get install python3 python3-dev python3-pip
```

#### Install software dependencies
```
sudo apt-get install libsm6 libxrender-dev \
    libboost-python-dev build-essential libssl-dev cmake \
    openmpi-bin openmpi-doc libopenmpi-dev
```

#### Install pip packages
```
sudo -H pip3 install --upgrade pip setuptools testresources scikit-build
sudo -H pip3 install --upgrade numpy scipy h5py colorama numpy-stl \
    numba imagecodecs-lite tifffile scikit-image opencv-python \
    Pillow nibabel medpy SimpleITK mpi4py itk vtk matplotlib
```

#### Download or clone Biomedisa
```
sudo apt-get install git
mkdir ~/git
cd ~/git
git clone https://github.com/biomedisa/biomedisa
```

#### Install CUDA 11.0
```
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
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
sudo -H "PATH=/usr/local/cuda-11.0/bin:${PATH}" pip3 install --upgrade pycuda

# Verify that PyCUDA is working properly
python3 ~/git/biomedisa/biomedisa_features/pycuda_test.py
```

#### Install TensorFlow
```
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
sudo apt-get update

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libnvinfer8_8.0.0-1+cuda11.0_amd64.deb
sudo apt install ./libnvinfer8_8.0.0-1+cuda11.0_amd64.deb
sudo apt-get update

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    libcudnn8=8.0.5.39-1+cuda11.0  \
    libcudnn8-dev=8.0.5.39-1+cuda11.0

# Install TensorRT. Requires that libcudnn8 is installed above.
sudo apt-get install --no-install-recommends libnvinfer8=8.0.0-1+cuda11.0 \
    libnvinfer-dev=8.0.0-1+cuda11.0 \
    libnvinfer-plugin8=8.0.0-1+cuda11.0

# Install TensorFlow
sudo -H pip3 install tensorflow-gpu==2.4.1
```
