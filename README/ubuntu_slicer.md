# Ubuntu 22/24 (3D Slicer extension)

- [Install Biomedisa](#install-biomedisa)
- [Clone the Biomedisa Repository](#clone-the-biomedisa-repository)
- [Download 3D Slicer](#download-3d-slicer)
- [Install pip packages](#install-pip-packages)
- [Add Biomedisa modules to 3D Slicer](#add-biomedisa-modules-to-3d-slicer)
- [Automatic Configuration](#automatic-configuration)
- [Manual Configuration (if needed)](#manual-configuration-if-needed)

#### Install Biomedisa
Install one of the Biomedisa [command-line based](https://github.com/biomedisa/biomedisa/#installation-command-line-based) versions.

#### Clone the Biomedisa Repository
```
mkdir ~/git
cd ~/git
git clone https://github.com/biomedisa/biomedisa.git
```

#### Download 3D Slicer
Download [3D Slicer](https://download.slicer.org/) and extract the files to a location of your choice.

#### Install pip packages using the Python environment in 3D Slicer
Run `PythonSlicer` from `Slicer-<VERSION>-linux-amd64/bin` (replace `<VERSION>` with your specific version):
```
cd Slicer-<VERSION>-linux-amd64/bin
./PythonSlicer -m pip install PyQt5 tifffile h5py
```

#### Add Biomedisa modules to 3D Slicer
Start 3D Slicer  
Edit -> Application Settings -> Modules  
Drag and Drop the following directories in the field "Additional module paths"  
Use only the first line if you only want to install the Smart Interpolation.
```
git/biomedisa/biomedisa_slicer_extension/biomedisa_extension/SegmentEditorBiomedisa
git/biomedisa/biomedisa_slicer_extension/biomedisa_extension/SegmentEditorBiomedisaPrediction
git/biomedisa/biomedisa_slicer_extension/biomedisa_extension/SegmentEditorBiomedisaTraining
```
Restart 3D Slicer.

#### Automatic Configuration
The modules will attempt to automatically locate your Biomedisa installation.
 
#### Manual Configuration (if needed)
If the automatic setup does not work because the location of your Python environment differs from the default setup, create a configuration file:
```
cd git/biomedisa/biomedisa_slicer_extension/biomedisa_extension
cp config_template.py config.py
```
Edit the `config.py` file to update the following paths based on your Biomedisa installation: **python_path**, **lib_path**, **wsl_path** (examples are provided in the `config_template.py` file).

