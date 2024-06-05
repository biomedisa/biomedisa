# Install Biomedisa from source
To develop Biomedisa or for the latest version, clone the repository and append its location to PYTHONPATH:

**Windows:** Download and install [Git](https://github.com/git-for-windows/git/releases/download/v2.45.1.windows.1/Git-2.45.1-64-bit.exe).  
**Ubuntu:**
```
sudo apt-get install git
```

#### Clone the Biomedisa repository
```
mkdir git
cd git
git clone https://github.com/biomedisa/biomedisa.git
```

#### Add the Biomedisa base directory (e.g., C:\Users\USERNAME\git\biomedisa) to the PYTHONPATH variable
**Windows:**  
Open Windows Search  
Type `View advanced system settings`  
Click `Environment Variables...`  
If not existing, create the **System variable** `PYTHONPATH` with value `C:\Users\USERNAME\git\biomedisa`  
Make sure to replace USERNAME with your name  
**Ubuntu:**
```
echo 'export PYTHONPATH=${HOME}/git/biomedisa:${PYTHONPATH}' >> ~/.bashrc
source ~/.bashrc
```

#### Biomedisa examples
Download test files from [Gallery](https://biomedisa.info/gallery/)
```
# smart interpolation
python -m biomedisa.interpolation Downloads/tumor.tif Downloads/labels.tumor.tif

# deep learning
python -m biomedisa.deeplearning Downloads/testing_axial_crop_pat13.nii.gz Downloads/heart.h5
```

#### Update Biomedisa
```
cd ~/git/biomedisa
git pull
```
