# Windows + Smart Interpolation + Deep Learning

- [Install GPU Driver](#install-nvidia-driver)
- [Install Anaconda3](#install-anaconda3)
- [Install Biomedisa Environment](#install-biomedisa-environment)
- [Download Test Files](#download-test-files)
- [Biomedisa Examples](#biomedisa-examples)
- [Update Biomedisa](#update-biomedisa)
- [Install Biomedisa from Source (Optional)](#install-biomedisa-from-source-optional)

#### Option 1: Install GPU Driver (NVIDIA, AMD, Intel)
Use Windows Search: `Check for updates` and `View optional updates`  
Windows automatically detects your GPU and installs the required drivers.  
Alternatively, install them manually, e.g. Download and install [NVIDIA](https://www.nvidia.com/Download/Find.aspx?lang=en-us) driver.  
Warning: Intel integrated GPUs (iGPUs) are not recommended, as they may produce incorrect results.

#### Option 2: Install CPU Runtime for OpenCL
For Intel CPU support, download and install [Intel CPU Runtime for OpenCL Applications for Windows OS](https://software.intel.com/en-us/articles/opencl-drivers).

#### Install Anaconda3
Download and install [Anaconda3](https://repo.anaconda.com/archive/).

#### Install Biomedisa Environment
Open Anaconda Prompt (e.g. Windows Search `Anaconda Prompt`). Download Biomedisa environment:
```
curl https://raw.githubusercontent.com/biomedisa/biomedisa/refs/heads/master/biomedisa_env.yml --output biomedisa_env.yml
```
Install Biomedisa environment:
```
conda env create --file biomedisa_env.yml
```
Note: If your computer didn't find `biomedisa_env.yml` the easiest way is to locate the file in your User directory and drag and drop it onto the Anaconda Prompt after typing `conda env create --file`.

#### Optional: Install Deep Learning (NVIDIA GPUs only)
Activate conda environment:
```
conda activate biomedisa
```
Install PyTorch as follows or check [PyTorch](https://pytorch.org/get-started/locally/):
```
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
Install Keras 3:
```
conda install conda-forge::keras
```

#### Download Test Files
Download test files from [Gallery](https://biomedisa.info/gallery/) or via command-line:
```
curl https://biomedisa.info/media/images/tumor.tif --output C:\Users\%USERNAME%\Downloads\tumor.tif
curl https://biomedisa.info/media/images/labels.tumor.nrrd --output C:\Users\%USERNAME%\Downloads\labels.tumor.nrrd
```
Deep learning test files:
```
curl https://biomedisa.info/media/images/mouse_molar_tooth.tif --output C:\Users\%USERNAME%\Downloads\mouse_molar_tooth.tif
curl https://biomedisa.info/media/images/teeth.h5 --output C:\Users\%USERNAME%\Downloads\teeth.h5
```

#### Biomedisa Examples
Activate conda environment:
```
conda activate biomedisa
```
Run the interpolation (the first run might take a bit longer):
```
python -m biomedisa.interpolation Downloads\tumor.tif Downloads\labels.tumor.nrrd
```
Specify the platform if the wrong platform is detected, e.g. `opencl_AMD_GPU` or `opencl_Intel_CPU`:
```
python -m biomedisa.interpolation Downloads\tumor.tif Downloads\labels.tumor.nrrd --platform=opencl_AMD_GPU
```
Test deep learning:
```
python -m biomedisa.deeplearning Downloads\mouse_molar_tooth.tif Downloads\teeth.h5
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

#### Install Biomedisa from Source (Optional)
To develop Biomedisa or for the latest version install Biomedisa from [source](https://github.com/biomedisa/biomedisa/blob/master/README/installation_from_source.md).

