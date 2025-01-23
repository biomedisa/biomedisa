# Windows 10/11 + Smart Interpolation + Deep Learning (command-line)

- [Install Microsoft Visual Studio 2022](#install-microsoft-visual-studio-2022)
- [Set Path Variable](#set-path-variable)
- [Install NVIDIA driver](#install-nvidia-driver)
- [Install CUDA Toolkit](#install-cuda-toolkit)
- [Install Microsoft MPI](#install-microsoft-mpi)
- [Install Anaconda3](#install-anaconda3)
- [Install Biomedisa environment](#install-biomedisa-environment)
- [Biomedisa examples](#biomedisa-examples)
- [Update Biomedisa](#update-biomedisa)
- [Remove Biomedisa environment](#remove-biomedisa-environment)
- [Install Biomedisa from source (optional)](#install-biomedisa-from-source-optional)

#### Install Microsoft Visual Studio 2022
Download and install [MS Visual Studio](https://visualstudio.microsoft.com/de/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&passive=false&cid=2030). This [YouTube tutorial](https://www.youtube.com/watch?v=Ia4cMBDJXrI) explains this step in more detail.
```
Select "Desktop development with C++"
Install
Restart Windows
```

#### Set Path Variable
**Option 1: Via [command-line](https://github.com/biomedisa/biomedisa/blob/master/README/deprecated/windows_path_powershell.md)**.  
**Option 2: Manually:**  
Step 1: Open PowerShell (e.g. Windows Search `PowerShell`) and get the Microsoft Visual Studio path using the following command:
```
Resolve-Path -Path "C:\Program Files\Microsoft Visual Studio\*\Community\VC\Tools\MSVC\*\bin\Hostx64\x64" | select -ExpandProperty Path
```
The output should look like `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64` but year `2022` and version number `14.37.32822` can be different in your case. Please use exactly the path from the output in *Step 4*.  
Step 2: Open Windows Search and type `View advanced system settings`  
Step 3: Click `Environment Variables...`  
Step 4: Add exactly the output from *Step 1* to the **System variable** `Path`

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

#### Install Anaconda3
Download and install [Anaconda3](https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Windows-x86_64.exe).

#### Install Biomedisa environment
Open Anaconda Prompt (e.g. Windows Search `Anaconda Prompt`). Download and install Biomedisa:
```
curl https://raw.githubusercontent.com/biomedisa/biomedisa/refs/heads/master/conda_environment.yml --output conda_environment.yml
conda env create -f conda_environment.yml
```
Note: If your computer didn't find `conda_environment.yml` the easiest way is to locate the file in your User directory and drag and drop it onto the Anaconda Prompt after typing `conda env create -f`.

#### Biomedisa Verification
Activate conda environment:
```
conda activate biomedisa
```
Verify that PyCUDA is working properly in the 3D Slicer environment:
```
python -m biomedisa.features.pycuda_test
```
Verify that TensorFlow detects your GPUs:
```
python -c "import tensorflow as tf; print('Detected GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

#### Biomedisa Examples
Download test files from [Gallery](https://biomedisa.info/gallery/) or via command-line:
```
curl https://biomedisa.info/media/images/tumor.tif --output C:\Users\%USERNAME%\Downloads\tumor.tif
curl https://biomedisa.info/media/images/labels.tumor.nrrd --output C:\Users\%USERNAME%\Downloads\labels.tumor.nrrd
curl https://biomedisa.info/media/images/testing_axial_crop_pat13.nii.gz --output C:\Users\%USERNAME%\Downloads\testing_axial_crop_pat13.nii.gz
curl https://biomedisa.info/media/images/heart.h5 --output C:\Users\%USERNAME%\Downloads\heart.h5
```
Smart Interpolation:
```
python -m biomedisa.interpolation Downloads\tumor.tif Downloads\labels.tumor.nrrd
```
Deep Learning:
```
python -m biomedisa.deeplearning Downloads\testing_axial_crop_pat13.nii.gz Downloads\heart.h5 -p
```

#### Install Biomedisa from source (optional)
To develop Biomedisa or for the latest version install Biomedisa from [source](https://github.com/biomedisa/biomedisa/blob/master/README/installation_from_source.md).

