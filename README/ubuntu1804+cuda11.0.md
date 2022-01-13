#  Ubuntu 18.04.5 LTS + CUDA 11.0 (full installation)

- [Install Python and pip](#install-python-and-pip)
- [Install software dependencies](#install-software-dependencies)
- [Install pip packages](#install-pip-packages)
- [Download or clone Biomedisa](#download-or-clone-biomedisa)
- [Install MySQL database](#install-mysql-database)
- [Install CUDA 11.0](#install-cuda-11.0)
- [Install TensorFlow](#install-tensorflow)
- [Run Biomedisa](#run-biomedisa)
- [Install Apache Server (optional)](#install-apache-server-optional)

#### Install Python and pip
```
sudo apt-get install python3 python3-dev python3-pip
```

#### Install software dependencies
```
sudo apt-get install libsm6 libxrender-dev libmysqlclient-dev \
    libboost-python-dev build-essential screen libssl-dev cmake \
    openmpi-bin openmpi-doc libopenmpi-dev redis-server
```

#### Install pip packages
```
sudo -H pip3 install --upgrade pip setuptools testresources scikit-build
sudo -H pip3 install --upgrade numpy scipy h5py colorama wget numpy-stl \
    numba imagecodecs-lite tifffile scikit-image opencv-python \
    Pillow nibabel medpy SimpleITK mpi4py itk vtk rq mysqlclient
sudo -H pip3 install django==3.2.6
```

#### Download or clone Biomedisa
```
sudo apt-get install git
mkdir ~/git
cd ~/git
git clone https://github.com/biomedisa/biomedisa
```

#### Adapt Biomedisa config
Make `config.py` as a copy of `config_example.py`
```
cp biomedisa/biomedisa_app/config_example.py biomedisa/biomedisa_app/config.py
```
In particular, adapt the following lines in `biomedisa/biomedisa_app/config.py`
```
'PATH_TO_BIOMEDISA' : '/home/dummy/git/biomedisa', # this is the path to your main biomedisa folder
'DJANGO_DATABASE' : 'biomedisa_user_password', # password for the user 'biomedisa' of your biomedisa_database (set up in the next step)
'ALLOWED_HOSTS' : ['YOUR_IP', 'localhost', '0.0.0.0'], # you must tell django explicitly which hosts are allowed (e.g. your IP or the URL of your homepage)
```

#### Install MySQL database
```
# Install MySQL
sudo apt-get install mysql-server

    *****************************
    set root password for MySQL database
    *****************************
    if password was not set during installation
    login as sudo and set your password manually
    *****************************
    sudo mysql -u root -p
    ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'biomedisa_root_password';
    quit;
    sudo service mysql stop
    sudo service mysql start
    *****************************

# Login to MySQL (with the password you just set)
mysql -u root -p

# Create a user 'biomedisa' with password 'biomedisa_user_password' (same as set for 'DJANGO_DATABASE')
GRANT ALL PRIVILEGES ON *.* TO 'biomedisa'@'localhost' IDENTIFIED BY 'biomedisa_user_password';

# Create Biomedisa database
CREATE DATABASE biomedisa_database;
exit;

# Add the following line to /etc/mysql/mysql.conf.d/mysqld.cnf
wait_timeout = 604800

# Migrate database and create superuser
cd ~/git/biomedisa
python3 manage.py migrate
python3 manage.py createsuperuser
```

#### Install CUDA 11.0
```
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get install --no-install-recommends cuda-11-0

# Reboot. Check that GPUs are visible using the command
nvidia-smi

# Add the following lines to your .bashrc (e.g. nano ~/.bashrc)
export CUDA_HOME=/usr/local/cuda-11.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

# Reload .bashrc and verify that CUDA is installed properly
source ~/.bashrc
nvcc --version

# Install PyCUDA
sudo -H PATH=/usr/local/cuda-11.0/bin:${PATH} pip3 install --upgrade pycuda

# Verify that PyCUDA is working properly
python3 ~/git/biomedisa/biomedisa_features/pycuda_test.py
```

#### Install TensorFlow
```
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt install ./libnvinfer7_7.1.3-1+cuda11.0_amd64.deb
sudo apt-get update

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    libcudnn8=8.0.4.30-1+cuda11.0  \
    libcudnn8-dev=8.0.4.30-1+cuda11.0

# Install TensorRT. Requires that libcudnn8 is installed above.
sudo apt-get install -y --no-install-recommends libnvinfer7=7.1.3-1+cuda11.0 \
    libnvinfer-dev=7.1.3-1+cuda11.0 \
    libnvinfer-plugin7=7.1.3-1+cuda11.0

# Install TensorFlow
sudo -H pip3 install tensorflow-gpu==2.4.1
```

#### Run Biomedisa
Start workers (this has to be done after each reboot)
```
cd ~/git/biomedisa
./start_workers.sh
```

Start Biomedisa locally
```
python3 manage.py runserver localhost:8080
```

#### Open Biomedisa
Open Biomedisa in your local browser http://localhost:8080/ and log in as the `superuser` you created.

#### Install Apache Server (optional)
Follow the [installation instructions](https://github.com/biomedisa/biomedisa/blob/master/README/APACHE_SERVER.md).
