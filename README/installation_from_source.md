# Install Biomedisa from source
To develop Biomedisa or for the latest version, clone the repository and append its location to PYTHONPATH:

#### Windows
Download and install [Git](https://github.com/git-for-windows/git/releases/download/v2.45.1.windows.1/Git-2.45.1-64-bit.exe).  

Clone the Biomedisa repository:
```
mkdir git
cd git
git clone https://github.com/biomedisa/biomedisa.git
```

Add the Biomedisa base directory (e.g., `C:\Users\USERNAME\git\biomedisa`) to the PYTHONPATH variable (replace USERNAME with your name):  
Open Windows Search  
Type `View advanced system settings`  
Click `Environment Variables...`  
If not existing, create the **System variable** `PYTHONPATH` with value `C:\Users\USERNAME\git\biomedisa`  

#### Ubuntu
Install Git:
```
sudo apt-get install git
```
Clone the Biomedisa repository:
```
mkdir git
cd git
git clone https://github.com/biomedisa/biomedisa.git
```
Add the Biomedisa base directory (e.g., `${HOME}/git/biomedisa`) to the PYTHONPATH variable:
```
echo 'export PYTHONPATH=${HOME}/git/biomedisa:${PYTHONPATH}' >> ~/.bashrc
source ~/.bashrc
```
