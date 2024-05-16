# Windows 10 + OpenCL + GPU (command-line-only)

- [Install NVIDIA driver](#install-nvidia-driver)
- [Install Microsoft MPI](#install-microsoft-mpi)
- [Install Git](#install-git)
- [Clone Biomedisa](#clone-biomedisa)
- [Install Anaconda3](#install-anaconda3)
- [Install Biomedisa environment](#install-biomedisa-environment)
- [Biomedisa examples](#biomedisa-examples)
- [Remove Biomedisa environment](#remove-biomedisa-environment)

#### Install NVIDIA Driver
Download and install [NVIDIA](https://www.nvidia.com/Download/Find.aspx?lang=en-us).  
Choose *Windows Driver Type:* Standard  
Choose *Recommended/Beta:* Studio Driver

#### Install Microsoft MPI
Download and install [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=57467).
```
Select "msmpisetup.exe"
```

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
Activate conda environment.
```
conda activate biomedisa
```
Download test files from [Gallery](https://biomedisa.info/gallery/) and run
```
# smart interpolation
python -m biomedisa.interpolation Downloads\tumor.tif Downloads\labels.tumor.tif --platform=opencl_NVIDIA_GPU

# deep learning
python -m biomedisa.deeplearning Downloads\testing_axial_crop_pat13.nii.gz Downloads\heart.h5 -p -bs=12
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
Remove Biomedisa conda directory (optional).
```
cd C:\Users\%USERNAME%\anaconda3\envs
rmdir /s /q biomedisa
```
