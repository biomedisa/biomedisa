# Deep Learning + Windows 10 + CUDA 11.3 (command-line-only)

- [Install NVIDIA driver](#install-nvidia-driver)
- [Install Anaconda3](#install-anaconda3)
- [Install conda packages](#install-conda-packages)
- [Download or clone Biomedisa](#download-or-clone-biomedisa)
- [Biomedisa example](#biomedisa-example)


#### Install NVIDIA Driver
Download and install [NVIDIA](https://www.nvidia.com/Download/Find.aspx?lang=en-us).  
Choose *Windows Driver Type:* Standard  
Choose *Recommended/Beta:* Studio Driver


#### Install Anaconda3
Download and install [Anaconda3](https://www.anaconda.com/products/individual#windows).


#### Install conda packages
Open Anaconda Prompt (e.g. Windows Search `Anaconda Prompt`).
```
conda create -n biomedisa python=3.9
conda activate biomedisa
conda install -c conda-forge numpy scipy colorama numba
conda install -c conda-forge imagecodecs-lite tifffile scikit-image opencv Pillow
conda install -c conda-forge nibabel medpy SimpleITK itk vtk numpy-stl matplotlib
conda install -c conda-forge cudatoolkit=11.3.1
conda install -c conda-forge cudnn
conda install -c conda-forge tensorflow-gpu
conda install -c anaconda h5py
```

#### Download or clone Biomedisa
Download and install [Git](https://github.com/git-for-windows/git/releases/download/v2.28.0.windows.1/Git-2.28.0-64-bit.exe).
```
mkdir git
cd git
git clone https://github.com/biomedisa/biomedisa
```

#### Biomedisa example
Download test files from [Gallery](https://biomedisa.de/gallery/).
```
python git\biomedisa\demo\biomedisa_deeplearning.py Downloads\testing_axial_crop_pat13.nii.gz Downloads\heart.h5 -p
```

