# Windows 10 + OpenCL + GPU (command-line-only)

- [Install NVIDIA Driver](#install-nvidia-driver)
- [Install Microsoft MPI](#install-microsoft-mpi)
- [Install Anaconda3](#install-anaconda3)
- [Install Biomedisa Environment](#install-biomedisa-environment)
- [Biomedisa Examples](#biomedisa-examples)
- [Update Biomedisa](#update-biomedisa)
- [Remove Biomedisa Environment](#remove-biomedisa-environment)
- [Install Biomedisa from Source (Optional)](#install-biomedisa-from-source-optional)

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
Download and install [Anaconda3](https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Windows-x86_64.exe).

#### Install Biomedisa Environment
Open Anaconda Prompt (e.g. Windows Search `Anaconda Prompt`). Download and install Biomedisa:
```
curl https://raw.githubusercontent.com/biomedisa/biomedisa/refs/heads/master/conda_environment.yml --output conda_environment.yml
conda env create -f conda_environment.yml
```
Note: If your computer didn't find `conda_environment.yml` the easiest way is to locate the file in your User directory and drag and drop it onto the Anaconda Prompt after typing `conda env create -f`.

#### Biomedisa Examples
Activate conda environment:
```
conda activate biomedisa
```
Download test files from [Gallery](https://biomedisa.info/gallery/) and run:
```
# smart interpolation
python -m biomedisa.interpolation Downloads\tumor.tif Downloads\labels.tumor.tif --platform=opencl_NVIDIA_GPU

# deep learning
python -m biomedisa.deeplearning Downloads\testing_axial_crop_pat13.nii.gz Downloads\heart.h5
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

#### Remove Biomedisa Environment
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

#### Install Biomedisa from Source (Optional)
To develop Biomedisa or for the latest version install Biomedisa from [source](https://github.com/biomedisa/biomedisa/blob/master/README/installation_from_source.md).
