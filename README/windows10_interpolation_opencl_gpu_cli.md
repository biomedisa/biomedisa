# Windows 10 + Smart Interpolation + OpenCL + GPU (command-line-only)

- [Install NVIDIA driver](#install-nvidia-driver)
- [Install Microsoft MPI](#install-microsoft-mpi)
- [Install Git](#install-git)
- [Clone Biomedisa](#clone-biomedisa)
- [Install Anaconda3](#install-anaconda3)
- [Install conda and pip packages](#install-conda-and-pip-packages)
- [Biomedisa example](#biomedisa-example)

#### Install NVIDIA Driver
Download and install [NVIDIA](https://www.nvidia.com/Download/Find.aspx?lang=en-us).  
Choose *Windows Driver Type:* Standard  
Choose *Recommended/Beta:* Studio Driver

#### Install Microsoft MPI
Download and install [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467).
```
Select "msmpisetup.exe"
```

#### Install Git
Download and install [Git](https://github.com/git-for-windows/git/releases/download/v2.28.0.windows.1/Git-2.28.0-64-bit.exe).

#### Clone Biomedisa
```
mkdir git
cd git
git clone https://github.com/biomedisa/biomedisa.git
```

#### Install Anaconda3
Download and install [Anaconda3](https://www.anaconda.com/products/individual#windows).

#### Install conda and pip packages
Open Anaconda Prompt (e.g. Windows Search `Anaconda Prompt`).
```
conda create -n biomedisa python=3.9
conda activate biomedisa
conda install -c conda-forge numpy scipy colorama numba
conda install -c conda-forge imagecodecs-lite tifffile scikit-image opencv=4.5.1 Pillow
conda install -c conda-forge nibabel medpy SimpleITK itk vtk numpy-stl matplotlib
conda install -c conda-forge cudatoolkit=11.3.1
pip install -U pyopencl mpi4py
```

#### Biomedisa example
Activate conda environment.
```
conda activate biomedisa
```
Download test files from [Gallery](https://biomedisa.de/gallery/) and run
```
python git\biomedisa\demo\biomedisa_interpolation.py Downloads\tumor.tif Downloads\labels.tumor.tif --platform opencl_NVIDIA_GPU
```

