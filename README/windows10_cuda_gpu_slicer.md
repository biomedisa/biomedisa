# Windows 10/11 + Smart Interpolation (3D Slicer extension)

- [Install Microsoft Visual Studio 2022](#install-microsoft-visual-studio-2022)
- [Set Path Variable](#set-path-variable-manually)
- [Install NVIDIA driver](#install-nvidia-driver)
- [Install CUDA Toolkit](#install-cuda-toolkit)
- [Install Microsoft MPI](#install-microsoft-mpi)
- [Install Git](#install-git)
- [Clone Biomedisa](#clone-biomedisa)
- [Install 3D Slicer](#install-3d-slicer)
- [Add Biomedisa module to 3D Slicer](#add-biomedisa-module-to-3d-slicer)
- [Install Anaconda3](#install-anaconda3)
- [Initialize PyCUDA installation](#initialize-pycuda-installation)
- [Install pip packages](#install-pip-packages)
- [Install PyCUDA](#install-pycuda)

#### Install Microsoft Visual Studio 2022
Download and install [MS Visual Studio](https://visualstudio.microsoft.com/de/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&passive=false&cid=2030). This [YouTube tutorial](https://www.youtube.com/watch?v=Ia4cMBDJXrI) explains this step in more detail.
```
Select "Desktop development with C++"
Install
Restart Windows
```

#### Set Path Variable
Open PowerShell (e.g. Windows Search `PowerShell`) and get the Microsoft Visual Studio path using the following command:
```
Resolve-Path -Path "C:\Program Files\Microsoft Visual Studio\*\Community\VC\Tools\MSVC\*\bin\Hostx64\x64" | select -ExpandProperty Path
```
Note: The output should look like `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64` but year `2022` and version number `14.37.32822` can be different in your case. Please use exactly the path from the output.

*Option 1: Manually*  
Open Windows Search  
Type `View advanced system settings`  
Click `Environment Variables...`  
Add exactly the path from the output to the **System variable** `Path`

*Option 2: Via [command-line]](https://github.com/biomedisa/biomedisa/blob/master/README/windows_path_powershell.md)*

#### Install NVIDIA Driver
Download and install [NVIDIA](https://www.nvidia.com/Download/Find.aspx?lang=en-us).  
Choose *Windows Driver Type:* Standard  
Choose *Recommended/Beta:* Studio Driver

#### Install CUDA Toolkit
Download and install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

#### Install Microsoft MPI
Download and install [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467).
```
Select "msmpisetup.exe"
```

#### Install Git
Download and install [Git](https://github.com/git-for-windows/git/releases/download/v2.45.1.windows.1/Git-2.45.1-64-bit.exe).

#### Clone Biomedisa
Open Command Prompt (e.g. Windows Search `cmd`) and clone Biomedisa:
```
mkdir git
cd git
git clone https://github.com/biomedisa/biomedisa.git
```

#### Install 3D Slicer
Download and install [3D Slicer](https://download.slicer.org/).

#### Add Biomedisa module to 3D Slicer
Start 3D Slicer  
Edit -> Application Settings -> Modules  
Drag and Drop the following directory in the field "Additional module paths"  
```
git/biomedisa/biomedisa_slicer_extension/biomedisa_extension/SegmentEditorBiomedisa
```
Restart 3D Slicer.

#### Install Anaconda3
Download and install [Anaconda3](https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Windows-x86_64.exe).

#### Initialize PyCUDA installation
Open Anaconda Prompt (e.g. Windows Search `Anaconda Prompt`) and run:
```
conda create -n slicer_extension python=3.9
conda activate slicer_extension
python -m pip install pycuda
```

#### Install pip packages using the Python environment in 3D Slicer
You need to run `PythonSlicer.exe` from within `AppData\Local\slicer.org\Slicer VERSION\bin`:
```
cd "AppData\Local\slicer.org\Slicer 5.6.2\bin"
PythonSlicer.exe -m pip install pip setuptools testresources scikit-build
PythonSlicer.exe -m pip install numpy scipy h5py colorama numpy-stl
PythonSlicer.exe -m pip install numba imagecodecs tifffile scikit-image opencv-python netCDF4 mrcfile
PythonSlicer.exe -m pip install Pillow nibabel medpy SimpleITK mpi4py itk vtk matplotlib biomedisa
PythonSlicer.exe -m pip install importlib_metadata PyQt5
```

#### Install PyCUDA in the 3D Slicer environment
Conda environment must still be activated.
```
PythonSlicer.exe -m pip install pycuda
```

#### Verify PyCUDA is working properly in the 3D Slicer environment
```
PythonSlicer.exe -m biomedisa.features.pycuda_test
```
