# Windows 10 + Smart Interpolation + OpenCL + CPU (command-line-only)

- [Install Microsoft MPI](#install-microsoft-mpi)
- [Install OpenCL Runtime](#install-opencl-runtime)
- [Install Anaconda3](#install-anaconda3)
- [Install conda and pip packages](#install-conda-and-pip-packages)
- [Download or clone Biomedisa](#download-or-clone-biomedisa)
- [Biomedisa example](#biomedisa-example)

#### Install Microsoft MPI
Download and install [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467).
```
Select "msmpisetup.exe" to download
Install
```

#### Install OpenCL Runtime
Download and install [OpenCL Runtime](https://software.intel.com/en-us/articles/opencl-drivers).

#### Install Anaconda3
Download and install [Anaconda3](https://www.anaconda.com/products/individual#windows).

#### Install conda and pip packages
Open Anaconda Prompt (e.g. Windows Search `Anaconda Prompt`).
```
conda create -n biomedisa python=3.9
conda activate biomedisa
conda install -c conda-forge numpy scipy colorama numba
conda install -c conda-forge imagecodecs-lite tifffile scikit-image opencv Pillow
conda install -c conda-forge nibabel medpy SimpleITK itk vtk numpy-stl matplotlib
pip install -U pyopencl mpi4py
```

#### Download or clone Biomedisa
Download and install [Git](https://github.com/git-for-windows/git/releases/download/v2.28.0.windows.1/Git-2.28.0-64-bit.exe).
```
mkdir git
cd git
git clone https://github.com/biomedisa/biomedisa
```

#### Biomedisa example
Download test files from [Gallery](https://biomedisa.de/gallery/).
```
python git\biomedisa\demo\biomedisa_interpolation.py Downloads\tumor.tif Downloads\labels.tumor.tif --platform opencl_Intel_CPU
```

