# Ubuntu 20.04 LTS + Smart Interpolation + OpenCL + CPU (command-line-only)

- [Install Python and pip](#install-python-and-pip)
- [Install software dependencies](#install-software-dependencies)
- [Install pip packages](#install-pip-packages)
- [Download or clone Biomedisa](#download-or-clone-biomedisa)
- [Install OpenCL Runtime](#install-opencl-runtime)
- [Install PyOpenCL](#install-pyopencl)
- [Biomedisa example](#biomedisa-example)

#### Install Python and pip
```
sudo apt-get install python3 python3-dev python3-pip
```

#### Install software dependencies
```
sudo apt-get install libsm6 libxrender-dev \
    libboost-python-dev build-essential libssl-dev cmake \
    openmpi-bin openmpi-doc libopenmpi-dev \
    ocl-icd-libopencl1 opencl-headers clinfo ocl-icd-opencl-dev lsb-core
```

#### Install pip packages
```
sudo -H pip3 install --upgrade pip setuptools testresources scikit-build wheel
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

#### Install OpenCL Runtime
Download and install [OpenCL Runtime](https://software.intel.com/en-us/articles/opencl-drivers).
```
tar -xzf l_opencl_p_VERSION.tgz
cd l_opencl_p_VERSION.tgz
sudo ./install.sh
```
Follow installation instructions (ignore "Missing optional prerequisites -- Unsupported OS").

#### Install PyOpenCL
```
sudo -H pip3 install --upgrade pyopencl
```

#### Biomedisa example
Download test files from [Gallery](https://biomedisa.de/gallery/).
```
python3 git/biomedisa/demo/biomedisa_interpolation.py Downloads/tumor.tif Downloads/labels.tumor.tif --platform opencl_Intel_CPU
```

