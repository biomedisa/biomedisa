# Smart Interpolation + Windows 10 + CUDA 11.6 (command-line-only)

- [Install Microsoft Visual Studio 2022](#install-microsoft-visual-studio-2022)
- [Set Path Variables](#set-path-variables)
- [Install NVIDIA driver](#install-nvidia-driver)
- [Install CUDA Toolkit 11.6](#install-cuda-toolkit-11.0)
- [Install Microsoft MPI](#install-microsoft-mpi)
- [Install Anaconda3](#install-anaconda3)
- [Install conda and pip packages](#install-conda-and-pip-packages)
- [Download or clone Biomedisa](#download-or-clone-biomedisa)
- [Verify that PyCUDA is working properly](#verify-that-pycuda-is-working-properly)
- [Biomedisa example](#biomedisa-example)

#### Install Microsoft Visual Studio 2022
Download and install [MS Visual Studio](https://visualstudio.microsoft.com/de/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&passive=false&cid=2030).
```
Select "Desktop development with C++"
Install
Restart Windows
```

#### Set Path Variables
Open Windows Search  
Type `View advanced system settings`  
Click `Environment Variables...`  
Add the following value to the **System variable** `Path`
```
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.31.31103\bin\Hostx64\x64
```

#### Install NVIDIA Driver
Download and install [NVIDIA](https://www.nvidia.com/Download/Find.aspx?lang=en-us).  
Choose *Windows Driver Type:* Standard  
Choose *Recommended/Beta:* Studio Driver

#### Install CUDA Toolkit 11.6
Download and install [CUDA Toolkit 11.6](https://developer.nvidia.com/cuda-downloads).

#### Install Microsoft MPI
Download and install [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467).
```
Select "msmpisetup.exe" to download
Install
```

#### Install Anaconda3
Download and install [Anaconda3](https://www.anaconda.com/products/individual#windows).

#### Install conda and pip packages
Open Anaconda Prompt (e.g. Windows Search `Anaconda Prompt`).
```
conda create -n biomedisa python=3.9
conda activate biomedisa
conda install -c conda-forge numpy scipy h5py colorama numba
conda install -c conda-forge imagecodecs-lite tifffile scikit-image opencv Pillow
conda install -c conda-forge nibabel medpy SimpleITK itk vtk numpy-stl matplotlib
pip install -U pycuda mpi4py
```

#### Download or clone Biomedisa
Download and install [Git](https://github.com/git-for-windows/git/releases/download/v2.28.0.windows.1/Git-2.28.0-64-bit.exe).
```
mkdir git
cd git
git clone https://github.com/biomedisa/biomedisa
```

#### Verify that PyCUDA is working properly
```
python git\biomedisa\biomedisa_features\pycuda_test.py
```

#### Biomedisa example
Download test files from [Gallery](https://biomedisa.de/gallery/).
```
python git\biomedisa\demo\biomedisa_interpolation.py Downloads\tumor.tif Downloads\labels.tumor.tif
```