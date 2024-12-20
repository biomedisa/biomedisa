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
python3.10 -m venv biomedisa_env
source biomedisa_env/bin/activate
```

#### Install Pip Packages
Add your local pip directory to the PATH variable:
```
echo 'export PATH=${HOME}/.local/bin:${PATH}' >> ~/.bashrc
```
Download list of requirements and install pip packages:
```
wget https://raw.githubusercontent.com/biomedisa/biomedisa/refs/heads/master/requirements.txt
python3.10 -m pip install -r requirements.txt
```

#### Verify that TensorFlow detects your GPUs
```
python3.10 -c "import tensorflow as tf; print('Detected GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

#### Biomedisa Example
Download test files from [Gallery](https://biomedisa.info/gallery/) or via command-line:
```
wget -P ~/Downloads/ https://biomedisa.info/media/images/mouse_molar_tooth.tif
wget -P ~/Downloads/ https://biomedisa.info/media/images/teeth.h5
```
Deep Learning:
```
python3.10 -m biomedisa.deeplearning ~/Downloads/mouse_molar_tooth.tif ~/Downloads/teeth.h5 --extension='.nrrd'
```

#### Install Biomedisa from source (optional)
To develop Biomedisa or for the latest version install Biomedisa from [source](https://github.com/biomedisa/biomedisa/blob/master/README/installation_from_source.md).

