# Windows 10 + OpenCL + CPU (command-line-only)

- [Install Microsoft Visual Studio 2022](#install-microsoft-visual-studio-2022)
- [Option 1: Set Path Variable manually](#option-1-set-path-variable-manually)
- [Option 2: Set Path Variable using PowerShell](#option-2-set-path-variable-using-powershell)
- [Install Microsoft MPI](#install-microsoft-mpi)
- [Install Intel CPU Runtime for OpenCL](#install-intel-cpu-runtime-for-opencl)
- [Install Anaconda3](#install-anaconda3)
- [Install Biomedisa environment](#install-biomedisa-environment)
- [Biomedisa examples](#biomedisa-examples)
- [Update Biomedisa](#update-biomedisa)
- [Remove Biomedisa environment](#remove-biomedisa-environment)

#### Install Microsoft Visual Studio 2022
Download and install [MS Visual Studio](https://visualstudio.microsoft.com/de/thank-you-downloading-visual-studio/?sku=Community&channel=Release&version=VS2022&source=VSLandingPage&passive=false&cid=2030). This [YouTube tutorial](https://www.youtube.com/watch?v=Ia4cMBDJXrI) explains this step in more detail.
```
Select "Desktop development with C++"
Install
Restart Windows
```

#### Option 1: Set Path Variable manually
Open PowerShell (e.g. Windows Search `PowerShell`) and get the Microsoft Visual Studio path using the following command:
```
Resolve-Path -Path "C:\Program Files\Microsoft Visual Studio\*\Community\VC\Tools\MSVC\*\bin\Hostx64\x64" | select -ExpandProperty Path
```
Note: The output should look like `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64` but year `2022` and version number `14.37.32822` can be different in your case. Please use exactly the path from the output.

Open Windows Search  
Type `View advanced system settings`  
Click `Environment Variables...`  
Add exactly the path from the output to the **System variable** `Path`

#### Option 2: Set Path Variable using PowerShell
Skip this step if you did it manually.
Open PowerShell as administrator (e.g. Windows Search `PowerShell`).
```
$currentPath = [System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::Machine)
$newPath = Resolve-Path -Path "C:\Program Files\Microsoft Visual Studio\*\Community\VC\Tools\MSVC\*\bin\Hostx64\x64" | select -ExpandProperty Path
$newPathValue = "$currentPath;$newPath"
[System.Environment]::SetEnvironmentVariable('PATH', $newPathValue, [System.EnvironmentVariableTarget]::Machine)
```

#### Install Microsoft MPI
Download and install [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467).
```
Select "msmpisetup.exe"
```

#### Install Intel CPU Runtime for OpenCL
Download and install [Intel CPU Runtime for OpenCL Applications 18.1 for Windows OS](https://software.intel.com/en-us/articles/opencl-drivers).

#### Install Anaconda3
Download and install [Anaconda3](https://www.anaconda.com/products/individual#windows).

#### Install Biomedisa environment
Download [Biomedisa environment](https://biomedisa.info/media/conda_environment.yml).
Open Anaconda Prompt (e.g. Windows Search `Anaconda Prompt`) and install Biomedisa:
```
conda env create -f C:\Users\%USERNAME%\Downloads\conda_environment.yml
```
Note: If your computer didn't find `conda_environment.yml` the easiest way is to locate the file in your Download directory and drag and drop it onto the Anaconda Prompt after typing `conda env create -f`.

#### Biomedisa examples
Activate conda environment:
```
conda activate biomedisa
```
Download test files from [Gallery](https://biomedisa.info/gallery/) and run:
```
# smart interpolation
python -m biomedisa.interpolation Downloads\tumor.tif Downloads\labels.tumor.tif --platform=opencl_Intel_CPU

# deep learning
python -m biomedisa.deeplearning Downloads\testing_axial_crop_pat13.nii.gz Downloads\heart.h5 -p -bs=12
```

#### Update Biomedisa
Activate conda environment:
```
conda activate biomedisa
```
Update Biomedisa package:
```
pip install -U biomedisa
```

#### Remove Biomedisa environment
Deactivate Biomedisa environment:
```
conda deactivate
```
Remove the Biomedisa environment:
```
conda remove --name biomedisa --all
```
Remove Biomedisa conda directory (optional):
```
cd C:\Users\%USERNAME%\anaconda3\envs
rmdir /s /q biomedisa
```
