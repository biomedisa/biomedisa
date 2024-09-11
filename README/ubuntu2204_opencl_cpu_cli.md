# Ubuntu 22.04 LTS + Smart Interpolation + OpenCL + CPU (command-line-only)

- [Install Python and pip](#install-python-and-pip)
- [Install software dependencies](#install-software-dependencies)
- [Intel CPU Runtime for OpenCL](#intel-cpu-runtime-for-opencl)
- [Install pip packages](#install-pip-packages)
- [Biomedisa example](#biomedisa-example)
- [Update Biomedisa](#update-biomedisa)
- [Install Biomedisa from source (optional)](#install-biomedisa-from-source-optional)

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

#### Add local pip directory to PATH variable
```
echo 'export PATH=${HOME}/.local/bin:${PATH}' >> ~/.bashrc
source ~/.bashrc
```

#### Install pip packages
```
pip install --upgrade pip setuptools testresources scikit-build wheel
pip install --upgrade numpy scipy h5py colorama numpy-stl \
    numba imagecodecs tifffile scikit-image opencv-python netCDF4 mrcfile \
    Pillow nibabel medpy SimpleITK mpi4py itk vtk matplotlib pyopencl biomedisa
pip install tensorflow==2.13.0
```

#### Biomedisa example
Download test files from [Gallery](https://biomedisa.info/gallery/).
```
# smart interpolation
python -m biomedisa.interpolation Downloads/tumor.tif Downloads/labels.tumor.tif --platform=opencl_Intel_CPU

# deep learning using stride_size=64 (less accuracy but faster)
python -m biomedisa.deeplearning Downloads/testing_axial_crop_pat13.nii.gz Downloads/heart.h5 -p --stride_size=64
```

#### Update Biomedisa
```
pip3 install -U biomedisa
```

#### Install Biomedisa from source (optional)
To develop Biomedisa or for the latest version install Biomedisa from [source](https://github.com/biomedisa/biomedisa/blob/master/README/installation_from_source.md).
