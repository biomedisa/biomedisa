# Ubuntu 22.04 LTS + CUDA 11.8 + GPU (command-line-only)

- [Install Python and pip](#install-python-and-pip)
- [Install software dependencies](#install-software-dependencies)
- [Install pip packages](#install-pip-packages)
- [Clone Biomedisa](#clone-biomedisa)
- [Install CUDA 11.8](#install-cuda-11.8)
- [Install TensorFlow (optional)](#install-tensorflow-optional)
- [Biomedisa examples](#biomedisa-examples)

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

#### Install TensorFlow (optional)
Only required if you want to use Deep Learning
```
# Install development and runtime libraries.
sudo apt-get install --no-install-recommends \
    libcudnn8=8.8.0.121-1+cuda11.8 \
    libcudnn8-dev=8.8.0.121-1+cuda11.8

# Install TensorRT. Requires that libcudnn8 is installed above.
sudo apt-get install --no-install-recommends libnvinfer8=8.5.3-1+cuda11.8 \
    libnvinfer-dev=8.5.3-1+cuda11.8 \
    libnvinfer-plugin8=8.5.3-1+cuda11.8

# OPTIONAL: hold packages to avoid crash after system update
sudo apt-mark hold libcudnn8 libcudnn8-dev libnvinfer-dev libnvinfer-plugin8 libnvinfer8 cuda-11-8

# Install TensorFlow
sudo -H pip3 install tensorflow==2.13.0
```

#### Biomedisa examples
Download test files from [Gallery](https://biomedisa.de/gallery/).
```
# smart interpolation
python3 git/biomedisa/demo/biomedisa_interpolation.py Downloads/tumor.tif Downloads/labels.tumor.tif

# deep learning
python3 git/biomedisa/demo/biomedisa_deeplearning.py Downloads/testing_axial_crop_pat13.nii.gz Downloads/heart.h5 -p -bs 12
```

