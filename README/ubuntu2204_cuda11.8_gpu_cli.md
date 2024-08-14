# Ubuntu 22.04 LTS + Smart Interpolation + Deep Learning (command-line-only)

- [Install Python and pip](#install-python-and-pip)
- [Install software dependencies](#install-software-dependencies)
- [Install CUDA 11.8](#install-cuda-11.8)
- [Install TensorFlow](#install-tensorflow-optional)
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

# Add the CUDA paths to your '~/.bashrc' file
echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64' >> ~/.bashrc
echo 'export PATH=${CUDA_HOME}/bin:${PATH}' >> ~/.bashrc

# Reload .bashrc and verify that CUDA is installed properly
source ~/.bashrc
nvcc --version
```

#### Install TensorFlow
```
# Install development and runtime libraries.
sudo apt-get install --no-install-recommends \
    libcudnn8=8.8.0.121-1+cuda11.8 \
    libcudnn8-dev=8.8.0.121-1+cuda11.8

# OPTIONAL: hold packages to avoid crash after system update
sudo apt-mark hold libcudnn8 libcudnn8-dev cuda-11-8
```

#### Adapt PATH variable
Add the local pip directory to the PATH variable:
```
echo 'export PATH=${HOME}/.local/bin:${PATH}' >> ~/.bashrc
source ~/.bashrc
```

#### Install pip packages
Download Biomedisa [dependencies](https://biomedisa.info/media/requirements.txt) and install packages:
```
wget https://biomedisa.info/media/requirements.txt
python3 -m pip install -r requirements.txt
```

#### Install PyCUDA
```
PATH=/usr/local/cuda-11.8/bin:${PATH} pip3 install pycuda==2022.2.2
```

#### Verify that PyCUDA is working properly
```
python3 -m biomedisa.features.pycuda_test
```

#### Biomedisa examples
Download test files from [Gallery](https://biomedisa.info/gallery/) and run:
```
# smart interpolation
python3 -m biomedisa.interpolation Downloads/tumor.tif Downloads/labels.tumor.tif

# deep learning
python3 -m biomedisa.deeplearning Downloads/testing_axial_crop_pat13.nii.gz Downloads/heart.h5 -p
```

#### Install Biomedisa from source (optional)
To develop Biomedisa or for the latest version install Biomedisa from [source](https://github.com/biomedisa/biomedisa/blob/master/README/installation_from_source.md).
