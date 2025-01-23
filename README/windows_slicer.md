# Windows 10/11 (3D Slicer extension)

- [Install Biomedisa](#install-biomedisa)
- [Install Git](#install-git)
- [Clone the Biomedisa Repository](#clone-the-biomedisa-repository)
- [Install 3D Slicer](#install-3d-slicer)
- [Install pip packages](#install-pip-packages)
- [Add Biomedisa module to 3D Slicer](#add-biomedisa-module-to-3d-slicer)
- [Automatic Configuration](#automatic-configuration)
- [Manual Configuration (if needed)](#manual-configuration-if-needed)

#### Install Biomedisa
Install Biomedisa [command-line based](https://github.com/biomedisa/biomedisa/blob/master/README/windows_wsl.md) for Windows.

#### Install Git
Download and install [Git](https://github.com/git-for-windows/git/releases/download/v2.45.1.windows.1/Git-2.45.1-64-bit.exe).

#### Clone the Biomedisa Repository
Open Command Prompt (e.g. Windows Search `cmd`) and clone Biomedisa:
```
mkdir git
cd git
git clone https://github.com/biomedisa/biomedisa.git
```

#### Install 3D Slicer
Download and install [3D Slicer](https://download.slicer.org/).

#### Install pip packages using the Python environment in 3D Slicer
You need to run `PythonSlicer.exe` from within `AppData\Local\slicer.org\Slicer VERSION\bin`:
```
cd "AppData\Local\slicer.org\Slicer 5.6.2\bin"
PythonSlicer.exe -m pip install PyQt5 tifffile
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
If the automatic setup does not work, create a configuration file:
```
cd git/biomedisa/biomedisa_slicer_extension/biomedisa_extension
xcopy config_template.py config.py
```
Edit the `config.py` file to update the following paths based on your Biomedisa installation: **python_path**, **lib_path**, **wsl_path** (examples are provided in the `config_template.py` file).

