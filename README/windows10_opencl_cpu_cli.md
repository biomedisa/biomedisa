# Windows 10 + OpenCL + CPU (command-line-only)

- [Install Microsoft MPI](#install-microsoft-mpi)
- [Install Intel CPU Runtime for OpenCL](#install-intel-cpu-runtime-for-opencl)
- [Install Git](#install-git)
- [Clone Biomedisa](#Clone-biomedisa)
- [Install Anaconda3](#install-anaconda3)
- [Install Biomedisa environment](#install-biomedisa-environment)
- [Biomedisa examples](#biomedisa-examples)
- [Remove Biomedisa environment](#remove-biomedisa-environment)

#### Install Microsoft MPI
Download and install [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467).
```
Select "msmpisetup.exe"
```

#### Install Intel CPU Runtime for OpenCL
Download and install [Intel CPU Runtime for OpenCL Applications 18.1 for Windows OS](https://software.intel.com/en-us/articles/opencl-drivers).

#### Install Git
Download and install [Git](https://github.com/git-for-windows/git/releases/download/v2.28.0.windows.1/Git-2.28.0-64-bit.exe).

#### Clone Biomedisa
Open Command Prompt (e.g. Windows Search `Command Prompt`).
```
mkdir git
cd git
git clone https://github.com/biomedisa/biomedisa.git
```

#### Install Anaconda3
Download and install [Anaconda3](https://www.anaconda.com/products/individual#windows).

#### Install Biomedisa environment
Open Anaconda Prompt (e.g. Windows Search `Anaconda Prompt`).
```
conda env create -f git\biomedisa\conda_environment.yml
```
Note: If your computer didn't find the path `git\biomedisa\conda_environment.yml` the easiest way is to locate the file and drag and drop it onto the Anaconda Prompt.

#### Biomedisa examples
Activate conda environment.
```
conda activate biomedisa
```
Download test files from [Gallery](https://biomedisa.de/gallery/) and run
```
# smart interpolation
python git\biomedisa\biomedisa_features\biomedisa_interpolation.py Downloads\tumor.tif Downloads\labels.tumor.tif --platform opencl_Intel_CPU

# deep learning
python git\biomedisa\biomedisa_features\biomedisa_deeplearning.py Downloads\testing_axial_crop_pat13.nii.gz Downloads\heart.h5 -p -bs 12
```

#### Remove Biomedisa environment
Deactivate Biomedisa environment.
```
conda deactivate
```
Remove the Biomedisa environment.
```
conda remove --name biomedisa --all
```
