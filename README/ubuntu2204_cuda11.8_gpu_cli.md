# Ubuntu 22.04 LTS + Smart Interpolation + Deep Learning (command-line-only)

- [Install Python and pip](#install-python-and-pip)
- [Install software dependencies](#install-software-dependencies)
- [Install CUDA 11.8](#install-cuda-11.8)
- [Install cuDNN](#install-cudnn)
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
Biomedisa's Deep Learning framework requires TensorFlow 2.13, which is compatible with CUDA 11.8 and cuDNN 8.8.0. Please ensure that you install these specific versions, as higher versions are not yet supported. Add NVIDIA package repositories:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
```
If the error `W: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg)` occurs:
```
sudo apt-key export 3BF863CC | sudo gpg --dearmour -o /etc/apt/trusted.gpg.d/cuda-tools.gpg
```
Install CUDA Toolkit:
```
sudo apt-get update
sudo apt-get install --no-install-recommends cuda-11-8
```
Reboot and check that your GPUs are visible using the following command:
```
nvidia-smi
```

#### Adapt PATH variables
Add the local pip directory to the PATH variable:
```
echo 'export PATH=${HOME}/.local/bin:${PATH}' >> ~/.bashrc
```
Add the CUDA paths:
```
echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64' >> ~/.bashrc
echo 'export PATH=${CUDA_HOME}/bin:${PATH}' >> ~/.bashrc
```
Reload .bashrc and verify that CUDA is installed properly:
```
source ~/.bashrc
nvcc --version
```

#### Install cuDNN
Install development and runtime libraries:
```
sudo apt-get install --no-install-recommends \
    libcudnn8=8.8.0.121-1+cuda11.8 \
    libcudnn8-dev=8.8.0.121-1+cuda11.8
```
OPTIONAL: hold packages to avoid crash after a system update:
```
sudo apt-mark hold libcudnn8 libcudnn8-dev cuda-11-8
```

#### Install pip packages
Download list of requirements and install pip packages:
```
wget https://raw.githubusercontent.com/biomedisa/biomedisa/refs/heads/master/requirements.txt
python3 -m pip install -r requirements.txt
PATH=/usr/local/cuda-11.8/bin:${PATH} python3 -m pip install pycuda
```

#### Verify that PyCUDA is working properly
```
python3 -m biomedisa.features.pycuda_test
```

#### Verify that TensorFlow detects your GPUs
```
python3 -c "import tensorflow as tf; print('Detected GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

#### Biomedisa Examples
Download test files from [Gallery](https://biomedisa.info/gallery/) or via command-line:
```
wget -P ~/Downloads/ https://biomedisa.info/media/images/tumor.tif
wget -P ~/Downloads/ https://biomedisa.info/media/images/labels.tumor.nrrd
wget -P ~/Downloads/ https://biomedisa.info/media/images/mouse_molar_tooth.tif
wget -P ~/Downloads/ https://biomedisa.info/media/images/teeth.h5
```
Smart Interpolation:
```
python3 -m biomedisa.interpolation ~/Downloads/tumor.tif ~/Downloads/labels.tumor.nrrd
```
Deep Learning:
```
python3 -m biomedisa.deeplearning ~/Downloads/mouse_molar_tooth.tif ~/Downloads/teeth.h5 --predict --extension='.nrrd'
```

#### Install Biomedisa from source (optional)
To develop Biomedisa or for the latest version install Biomedisa from [source](https://github.com/biomedisa/biomedisa/blob/master/README/installation_from_source.md).

