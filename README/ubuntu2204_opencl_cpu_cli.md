# Ubuntu 22.04 LTS + Smart Interpolation + OpenCL + CPU (command-line-only)

- [Install Python and pip](#install-python-and-pip)
- [Install software dependencies](#install-software-dependencies)
- [Intel CPU Runtime for OpenCL](#intel-cpu-runtime-for-opencl)
- [Install pip packages](#install-pip-packages)
- [Biomedisa example](#biomedisa-example)
- [Update Biomedisa](#update-biomedisa)

#### Install Python and pip
```
sudo apt-get install python3 python3-dev python3-pip
```

#### Install software dependencies
```
sudo apt-get install libsm6 libxrender-dev unzip \
    libboost-python-dev build-essential libssl-dev cmake \
    openmpi-bin openmpi-doc libopenmpi-dev libgl1 \
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
```
pip3 install --upgrade pip setuptools testresources scikit-build wheel
pip3 install --upgrade numpy scipy h5py colorama numpy-stl \
    numba imagecodecs tifffile scikit-image opencv-python netCDF4 mrcfile \
    Pillow nibabel medpy SimpleITK mpi4py itk vtk matplotlib pyopencl biomedisa

# Add 'export PATH=${HOME}/.local/bin:${PATH}' to '~/.bashrc'
echo 'export PATH=${HOME}/.local/bin:${PATH}' >> ~/.bashrc
```

#### Biomedisa example
Download test files from [Gallery](https://biomedisa.info/gallery/).
```
python3 -m biomedisa.interpolation Downloads/tumor.tif Downloads/labels.tumor.tif --platform=opencl_Intel_CPU
```

#### Update Biomedisa
```
pip3 install -U biomedisa
```
