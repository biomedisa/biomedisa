# Windows 10 + CUDA 11.0 (command-line-only)

- [Install Microsoft Visual Studio 2017](#install-microsoft-visual-studio-2017)
- [Set Path Variables](#set-path-variables)
- [Install Python and pip](#install-python-and-pip)
- [Install pip packages](#install-pip-packages)
- [Install NVIDIA driver](#install-nvidia-driver)
- [Install CUDA Toolkit 11.0](#install-cuda-toolkit-11.0)
- [Disable TDR in Nsight Monitor](#disable-tdr-in-nsight-monitor)
- [Install Microsoft MPI](#install-microsoft-mpi)
- [Install PyCUDA and mpi4py](#install-pycuda-and-mpi4py)
- [Download or clone Biomedisa](#download-or-clone-biomedisa)
- [Verify that PyCUDA is working properly](#verify-that-pycuda-is-working-properly)
- [Biomedisa AI (optional)](#biomedisa-ai-optional)

#### Install Microsoft Visual Studio 2017
Download and install [MS Visual Studio](https://visualstudio.microsoft.com/de/thank-you-downloading-visual-studio/?sku=Community&rel=15&rr=https%3A%2F%2Fwww.wintotal.de%2Fdownload%2Fmicrosoft-visual-studio-community-2017%2F).
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
C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64
```

#### Install Python and pip
Download and install [Python](https://www.python.org/downloads/windows/).
```
Choose "Latest Python 3 Release"
Scroll to "Files"
Choose "Windows x86-64 executable installer"
Select "Add Python 3.X to PATH"
Install
Disable path length limit
Close
```

#### Install pip packages
Open Command Prompt (e.g. Windows Search `Command Prompt`).
```
pip3 install --upgrade pip pypiwin32 setuptools wheel numpy scipy h5py colorama numba
pip3 install --upgrade imagecodecs-lite tifffile scikit-image opencv-python Pillow
pip3 install --upgrade nibabel medpy SimpleITK itk vtk numpy-stl matplotlib
```

#### Install NVIDIA Driver
Download and install [NVIDIA](https://www.nvidia.com/Download/Find.aspx?lang=en-us).  
Choose *Windows Driver Type:* Standard  
Choose *Recommended/Beta:* Studio Driver

#### Install CUDA Toolkit 11.0
Download and install [CUDA Toolkit 11.0](https://developer.nvidia.com/cuda-downloads).

#### Disable TDR in Nsight Monitor
```
Windows Search "Nsight Monitor" (run as administrator)
Click the "NVIDIA Nsight" symbol in the right corner of your menu bar
Click Nsight Monitor options
Disable "WDDM TDR Enabled"
Reboot your system
```

#### Install Microsoft MPI
Download and install [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467).
```
Select "msmpisetup.exe" to download
Install
```

#### Install PyCUDA and mpi4py
Open Command Prompt (e.g. Windows Search `Command Prompt`).
```
pip3 install --upgrade pycuda mpi4py
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
python biomedisa/biomedisa_features/pycuda_test.py
```

#### Biomedisa AI (optional)

##### Install cuDNN
Download [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) (free NVIDIA account required).  
Extract the ZIP folder.

##### Set Path Variables
Open Windows Search  
Type `View advanced system settings`  
Click `Environment Variables...`  
Add the following value to the **System variable** `Path`
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\extras\CUPTI\lib64
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include
C:\Users\USERNAME\cuda\bin      (The path where you extracted cuDNN)
```

##### Install TensorFlow
```
pip3 install tensorflow-gpu==2.4.1
```
