# Ubuntu 22.04 LTS + CUDA 11.8 + GPU (3D Slicer extension)

- [Download 3D Slicer](#download-3d-slicer)
- [Install software dependencies](#install-software-dependencies)
- [Clone Biomedisa](#clone-biomedisa)
- [Add Biomedisa modules to 3D Slicer](#add-biomedisa-modules-to-3d-slicer)
- [Install pip packages](#install-pip-packages)
- [Install CUDA 11.8](#install-cuda-11.8)
- [Install cuDNN (optional)](#install-cudnn-optional)
- [Install mpi4py, PyCUDA, and TensorFlow](#install-mpi4py,-pycuda,-and-tensorflow)

#### Download 3D Slicer
Download [3D Slicer](https://download.slicer.org/) and extract the files to a location of your choice.

#### Install software dependencies
```
sudo apt-get install libsm6 libxrender-dev unzip \
    libboost-python-dev build-essential libssl-dev cmake \
    openmpi-bin openmpi-doc libopenmpi-dev git libgl1
```

#### Clone Biomedisa
```
mkdir ~/git
cd ~/git
git clone https://github.com/biomedisa/biomedisa.git
```

#### Add Biomedisa modules to 3D Slicer
Start 3D Slicer  
Edit -> Application Settings -> Modules  
Drag and Drop the following directories in the field "Additional module paths"  
Use only the first line if you only want to install Smart Interpolation.
```
git/biomedisa/biomedisa_slicer_extension/biomedisa_extension/SegmentEditorBiomedisa
git/biomedisa/biomedisa_slicer_extension/biomedisa_extension/SegmentEditorBiomedisaPrediction
git/biomedisa/biomedisa_slicer_extension/biomedisa_extension/SegmentEditorBiomedisaTraining
```
Restart 3D Slicer.

#### Install pip packages using the Python environment in 3D Slicer
You need to run `PythonSlicer` from within `Slicer-VERSION-linux-amd64/bin`:
```
./PythonSlicer -m pip install pip setuptools testresources scikit-build
./PythonSlicer -m pip install numpy scipy h5py colorama numpy-stl \
    numba imagecodecs tifffile scikit-image opencv-python netCDF4 mrcfile \
    Pillow nibabel medpy SimpleITK itk vtk matplotlib biomedisa \
    importlib_metadata PyQt5
```

#### Install CUDA 11.8
Biomedisa's Deep Learning framework requires TensorFlow 2.13, which is compatible with CUDA 11.8 and cuDNN 8.8.0. Please ensure that you install these specific versions, as higher versions are not yet supported. You may choose any CUDA version compatible with your NVIDIA GPU architecture as outlined in the [NVIDIA Documentation](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html) if you skip the cuDNN installation for the Deep Learning module below. Add NVIDIA package repositories:
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
Add the CUDA paths to your '~/.bashrc' file:
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

#### Install cuDNN (optional)
Only required if you want to use Deep Learning. Install development and runtime libraries:
```
sudo apt-get install --no-install-recommends \
    libcudnn8=8.8.0.121-1+cuda11.8 \
    libcudnn8-dev=8.8.0.121-1+cuda11.8
```
OPTIONAL: hold packages to avoid crash after a system update:
```
sudo apt-mark hold libcudnn8 libcudnn8-dev cuda-11-8
```

#### Install Python 3.9 and development tools
3D Slicer uses Python 3.9. To install mpi4py, PyCUDA, and TensorFlow, ensure you have Python 3.9 and its development tools installed:
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.9 python3.9-dev
sudo apt install python3.9-distutils
```

#### Install mpi4py, PyCUDA, and TensorFlow
**Option 1:** Install the following PIP packages for Python 3.9:
```
python3.9 -m pip install mpi4py
PATH=/usr/local/cuda-11.8/bin:${PATH} python3.9 -m pip install pycuda
python3.9 -m pip install tensorflow==2.13.0
```
Add local pip directory to PATH variable:
```
echo 'export PATH=${HOME}/.local/bin:${PATH}' >> ~/.bashrc
source ~/.bashrc
```
Install the PIP packages into the 3D Slicer environment:
```
./PythonSlicer -m pip install mpi4py
PATH=/usr/local/cuda-11.8/bin:${PATH} ./PythonSlicer -m pip install pycuda
./PythonSlicer -m pip install tensorflow==2.13.0
```
**Option 2 (only mpi4py and PyCUDA):** Direct installation in the 3D Slicer environment:
```
./PythonSlicer -c "import os,sys; os.environ['C_INCLUDE_PATH'] = '/usr/include/python3.9'; os.system(f'{sys.executable} -m pip install --no-cache-dir mpi4py')"
./PythonSlicer -c "import os,sys; os.environ['C_INCLUDE_PATH'] = '/usr/include/python3.9'; os.environ['CPLUS_INCLUDE_PATH'] = '/usr/include/python3.9'; os.environ['CC'] = '/usr/bin/gcc'; os.environ['CXX'] = '/usr/bin/g++'; os.system(f'{sys.executable} -m pip install --no-cache-dir pycuda')"
```

#### Verify that PyCUDA is working properly in the 3D Slicer environment
```
./PythonSlicer -m biomedisa.features.pycuda_test
```

#### Verify that TensorFlow detects your GPUs
```
./PythonSlicer -c "import tensorflow as tf; print('Detected GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

