# Ubuntu 22.04 LTS + Smart Interpolation + OpenCL + CPU (command-line-only)

- [Install Python and pip](#install-python-and-pip)
- [Install software dependencies](#install-software-dependencies)
- [Intel CPU Runtime for OpenCL](#intel-cpu-runtime-for-opencl)
- [Install pip packages](#install-pip-packages)
- [Clone Biomedisa](#clone-biomedisa)
- [Biomedisa example](#biomedisa-example)

#### Install Python and pip
```
sudo apt-get install python3 python3-dev python3-pip
```

#### Install software dependencies
```
sudo apt-get install libsm6 libxrender-dev \
    libboost-python-dev build-essential libssl-dev cmake \
    openmpi-bin openmpi-doc libopenmpi-dev git libgl1 \
    ocl-icd-libopencl1 opencl-headers clinfo ocl-icd-opencl-dev lsb-core
```

#### Intel CPU Runtime for OpenCL
Download and install [Intel CPU Runtime for OpenCL Applications 18.1 for Linux OS](https://software.intel.com/en-us/articles/opencl-drivers).
```
tar -xzf l_opencl_p_VERSION.tgz
cd l_opencl_p_VERSION
sudo ./install.sh
```
Follow installation instructions (ignore "Missing optional prerequisites -- Unsupported OS").

#### Install pip packages
You may only use `pip3 install --upgrade <package>` and add `export PATH=/home/$USER/.local/bin:${PATH}` to `~/.bashrc` if you only install Biomedisa for your user and do not have sudo rights, for example like on a supercomputer
```
sudo -H pip3 install --upgrade pip setuptools testresources scikit-build wheel
sudo -H pip3 install --upgrade numpy scipy h5py colorama numpy-stl \
    numba imagecodecs tifffile scikit-image opencv-python netCDF4 mrcfile \
    Pillow nibabel medpy SimpleITK mpi4py itk vtk matplotlib pyopencl
```

#### Clone Biomedisa
```
mkdir ~/git
cd ~/git
git clone https://github.com/biomedisa/biomedisa.git
```

#### Biomedisa example
Download test files from [Gallery](https://biomedisa.de/gallery/).
```
python3 git/biomedisa/demo/biomedisa_interpolation.py Downloads/tumor.tif Downloads/labels.tumor.tif --platform opencl_Intel_CPU
```

