# Windows 10 + Deep Learning (command-line-only)

- [Install NVIDIA driver](#install-nvidia-driver)
- [Install Git](#install-git)
- [Clone Biomedisa](#clone-biomedisa)
- [Install Anaconda3](#install-anaconda3)
- [Install conda environment](#install-conda-environment)
- [Biomedisa example](#biomedisa-example)


#### Install NVIDIA Driver
Download and install [NVIDIA](https://www.nvidia.com/Download/Find.aspx?lang=en-us).  
Choose *Windows Driver Type:* Standard  
Choose *Recommended/Beta:* Studio Driver

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

#### Install conda environment
Open Anaconda Prompt (e.g. Windows Search `Anaconda Prompt`).
```
conda env create -f git\biomedisa\conda_environment.yml
```

#### Biomedisa example
Activate conda environment.
```
conda activate biomedisa
```
Download test files from [Gallery](https://biomedisa.de/gallery/) and run
```
python git\biomedisa\demo\biomedisa_deeplearning.py Downloads\testing_axial_crop_pat13.nii.gz Downloads\heart.h5 -p -bs 12
```

