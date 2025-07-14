# Windows + OpenCL + GPU (command-line-only)

- [Install GPU Driver](#install-nvidia-driver)
- [Install Anaconda3](#install-anaconda3)
- [Install Biomedisa Environment](#install-biomedisa-environment)
- [Download Test Files](#download-test-files)
- [Biomedisa Examples](#biomedisa-examples)
- [Update Biomedisa](#update-biomedisa)
- [Install Biomedisa from Source (Optional)](#install-biomedisa-from-source-optional)

#### Install GPU Driver (Intel, AMD, NVIDIA)
Use Windows Search: `Check for updates` and `View optional updates`  
Windows automatically detects your GPU and installs the required drivers.  
Alternatively, install them manually, e.g. Download and install [NVIDIA](https://www.nvidia.com/Download/Find.aspx?lang=en-us).

#### Install Anaconda3
Download and install [Anaconda3](https://repo.anaconda.com/archive/).

#### Install Biomedisa Environment
Open Anaconda Prompt (e.g. Windows Search `Anaconda Prompt`). Download Biomedisa environment:
```
curl https://raw.githubusercontent.com/biomedisa/biomedisa/refs/heads/master/conda_interpolation.yml --output conda_interpolation.yml
```
Install Biomedisa environment:
```
conda env create --file conda_interpolation.yml
```
Note: If your computer didn't find `conda_interpolation.yml` the easiest way is to locate the file in your User directory and drag and drop it onto the Anaconda Prompt after typing `conda env create --file`.


#### Download Test Files
Download test files from [Gallery](https://biomedisa.info/gallery/) or via command-line:
```
curl https://biomedisa.info/media/images/tumor.tif --output C:\Users\%USERNAME%\Downloads\tumor.tif
curl https://biomedisa.info/media/images/labels.tumor.nrrd --output C:\Users\%USERNAME%\Downloads\labels.tumor.nrrd
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
Specify the platform if the wrong platform is detected:
```
python -m biomedisa.interpolation Downloads\tumor.tif Downloads\labels.tumor.nrrd --platform=opencl_AMD_GPU
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

