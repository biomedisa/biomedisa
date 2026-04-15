# Ubuntu 24.04 LTS + Smart Interpolation + Deep Learning (command-line)

- [Install Python and Pip](#install-python-and-pip)
- [Install Software Dependencies](#install-software-dependencies)
- [Install CUDA Toolkit](#install-cuda-toolkit)
- [Install Pip Packages](#install-pip-packages)
- [Install TensorFlow or PyTorch](#install-tensorflow-or-pytorch)
- [Biomedisa Examples](#biomedisa-examples)
- [Install Biomedisa from source (optional)](#install-biomedisa-from-source-optional)

#### Install Python and Pip
```
sudo apt-get install python3 python3.12-dev python3-pip python3-venv
```

#### Install Software Dependencies
```
sudo apt-get install libsm6 libxrender-dev unzip \
    libboost-python-dev build-essential libssl-dev cmake \
    openmpi-bin openmpi-doc libopenmpi-dev libgl1 wget
```

#### Install CUDA Toolkit
Download and install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) or via command-line as follows. You may choose any CUDA version compatible with your NVIDIA GPU architecture as outlined in the [NVIDIA Documentation](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html). If you select a version other than CUDA 12.6 for Ubuntu 22.04, you will need to adjust the following steps accordingly:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```
Reboot and check that your GPUs are visible using the following command:
```
nvidia-smi
```
Add the CUDA paths (adjust the CUDA version if required):
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

#### Create a virtual Python Environment
```
python3 -m venv biomedisa_env
source biomedisa_env/bin/activate
```

#### Install Pip Packages
Download the list of requirements manually from [GitHub](https://github.com/biomedisa/biomedisa/) or as follows:
```
wget https://raw.githubusercontent.com/biomedisa/biomedisa/refs/heads/master/requirements.txt
```

#### Install TensorFlow or PyTorch
- Choose one of the following frameworks:

TensorFlow (NVIDIA/CUDA):
```
python3 -m pip install -r requirements.txt keras tensorflow[and-cuda]==2.18.0 tf-keras==2.18.0
```
TensorFlow Blackwell GPUs (NVIDIA/CUDA):
```
python3 -m pip install -r requirements.txt keras tf-nightly[and-cuda]==2.21.0.dev20260203 tf-keras==2.21.0
```
PyTorch (NVIDIA/CUDA):
```
python3 -m pip install -r requirements.txt keras torch torchvision --extra-index-url https://download.pytorch.org/whl/cu126
```
TensorFlow (AMD/ROCm):
```
python3 -m pip install -r requirements.txt keras tf-keras==2.16 tensorflow-rocm==2.16.2 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/ --upgrade
```
PyTorch (AMD/ROCm):
```
python3 -m pip install -r requirements.txt keras torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm6.4
```
If you use PyTorch, you need to change the backend entry in `~/.keras/keras.json` to `torch`. PyTorch currently lacks support for auto-cropping and multi-GPU training.

#### Install PyCUDA
Adjust the CUDA version if required:
```
PATH=/usr/local/cuda-12.6/bin:${PATH} python3 -m pip install pycuda
```

#### Verify that PyCUDA is working properly
```
python3 -m biomedisa.features.pycuda_test
```

#### Verify that TensorFlow or PyTorch detects your GPUs
TensorFlow:
```
python3 -c "import tensorflow as tf; print('Detected GPUs:', len(tf.config.list_physical_devices('GPU')))"
```
PyTorch:
```
python3 -c "import torch; print('Detected GPUs:', torch.cuda.device_count())"
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
python3 -m biomedisa.deeplearning ~/Downloads/mouse_molar_tooth.tif ~/Downloads/teeth.h5 --extension='.nrrd'
```

#### Install Biomedisa from source (optional)
To develop Biomedisa or for the latest version install Biomedisa from [source](https://github.com/biomedisa/biomedisa/blob/master/README/installation_from_source.md).

