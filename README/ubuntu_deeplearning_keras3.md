# Ubuntu 22/24 LTS + Deep Learning (command-line)

- [Install Python 3.10 and Pip](#install-python-and-pip)
- [Install Software Dependencies](#install-software-dependencies)
- [Install Pip Packages](#install-pip-packages)
- [Biomedisa Examples](#biomedisa-examples)
- [Install Biomedisa from source (optional)](#install-biomedisa-from-source-optional)

#### Install Python 3.10 and Pip
Ubuntu 22.04:
```
sudo apt-get install python3 python3-dev python3-pip python3-venv
```
Ubuntu 24.04:
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.10 python3.10-dev python3-pip python3.10-venv
```

#### Install Software Dependencies
```
sudo apt-get install libsm6 libxrender-dev unzip \
    libboost-python-dev build-essential libssl-dev cmake \
    openmpi-bin openmpi-doc libopenmpi-dev libgl1 wget
```

#### Create a virtual Python Environment
```
python3.10 -m venv ~/biomedisa_env
source ~/biomedisa_env/bin/activate
```

#### Install Pip Packages
Download the list of requirements and install pip packages:
```
wget https://raw.githubusercontent.com/biomedisa/biomedisa/refs/heads/master/requirements.txt
python3.10 -m pip install -r requirements.txt
```

#### Install TensorFlow or PyTorch
- Choose one of the following frameworks:

TensorFlow (NVIDIA/CUDA):
```
python3.10 -m pip install keras tensorflow[and-cuda]==2.16.2 tf-keras==2.16
```
PyTorch (NVIDIA/CUDA):
```
python3.10 -m pip install keras torch torchvision --extra-index-url https://download.pytorch.org/whl/cu126
```
TensorFlow (AMD/ROCm):
```
python3.10 -m pip install keras tf-keras==2.16 tensorflow-rocm==2.16.2 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/ --upgrade
```
PyTorch (AMD/ROCm):
```
python3.10 -m pip install keras torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm6.4
```
If you use PyTorch, you need to change the backend entry in `~/.keras/keras.json` to `torch`. PyTorch currently lacks support for auto-cropping and multi-GPU training.

#### Verify that your GPUs are detected
TensorFlow:
```
python3.10 -c "import tensorflow as tf; print('Detected GPUs:', len(tf.config.list_physical_devices('GPU')))"
```
PyTorch:
```
python3.10 -c "import torch; print('Detected GPUs:', torch.cuda.device_count())"
```

#### Biomedisa Example
Download test files via command-line:
```
wget -P ~/Downloads/ https://biomedisa.info/media/images/mouse_molar_tooth.tif
wget -P ~/Downloads/ https://biomedisa.info/media/images/teeth.h5
```
Biomedisa inference test:
```
python3.10 -m biomedisa.deeplearning ~/Downloads/mouse_molar_tooth.tif ~/Downloads/teeth.h5 --extension='.nrrd'
```

#### Install Biomedisa from source (optional)
To develop Biomedisa or for the latest version install Biomedisa from [source](https://github.com/biomedisa/biomedisa/blob/master/README/installation_from_source.md).

