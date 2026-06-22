# Ubuntu 24.04 LTS + Deep Learning (command-line)

- [Install Python and Pip](#install-python-and-pip)
- [Install Software Dependencies](#install-software-dependencies)
- [Install Pip Packages](#install-pip-packages)
- [Install TensorFlow or PyTorch](#install-tensorflow-or-pytorch)
- [Biomedisa Example](#biomedisa-example)
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
- Choose one of the following frameworks. PyTorch currently lacks support for auto-cropping and multi-GPU training:

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
python3 -m pip install -r requirements.txt
python3 -m pip install tensorflow-rocm==2.18.1 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/ --upgrade
python3 -m pip install keras
python3 -m pip install tf-keras==2.18.0 --no-deps
```
PyTorch (AMD/ROCm):
```
python3 -m pip install -r requirements.txt keras torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm7.2
```

#### Optional: Install Segment Anything Model (SAM) for instance segmentation (requires PyTorch)
```
python3 -m pip install git+https://github.com/facebookresearch/segment-anything.git
```

#### Verify that your GPUs are detected
TensorFlow:
```
python3 -c "import tensorflow as tf; print('Detected GPUs:', len(tf.config.list_physical_devices('GPU')))"
```
PyTorch:
```
python3 -c "import torch; print('Detected GPUs:', torch.cuda.device_count())"
```

#### Biomedisa Example
Download test files via command-line:
```
wget -P ~/Downloads/ https://biomedisa.info/media/images/mouse_molar_tooth.tif
wget -P ~/Downloads/ https://biomedisa.info/media/images/teeth.h5
```
Biomedisa inference test:
```
python3 -m biomedisa.deeplearning ~/Downloads/mouse_molar_tooth.tif ~/Downloads/teeth.h5 --extension='.nrrd'
```

#### Install Biomedisa from source (optional)
To develop Biomedisa or for the latest version install Biomedisa from [source](https://github.com/biomedisa/biomedisa/blob/master/README/installation_from_source.md).

