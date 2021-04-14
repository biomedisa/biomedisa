#  Installation instructions (Ubuntu 18.04.5 LTS)

- [Install Python and pip](#nstall-python-and-pip)
- [Install software dependencies](#Install-software-dependencies)
- [Install pip packages](#install-pip-packages)
- [Download or clone Biomedisa](#download-or-clone-biomedisa)
- [Install MySQL database](#install-mysql-database)
- [Setting up CUDA environment](#setting-up-cuda-environment)
- [Install Tensorflow and Keras](#install-tensorflow-and-keras)
- [Run Biomedisa](#run-biomedisa)
- [Install Apache Server](#install-apache-server)

### Install Python and pip
```
sudo apt-get install python3 python3-dev python3-pip
```

### Install software dependencies
```
sudo apt-get install libsm6 libxrender-dev libmysqlclient-dev \
    libboost-python-dev build-essential screen libssl-dev cmake \
    openmpi-bin openmpi-doc libopenmpi-dev redis-server
```

### Install pip packages
```
sudo -H pip3 install --upgrade pip setuptools scikit-build 
sudo -H pip3 install --upgrade numpy scipy h5py colorama itk vtk wget \
    numba imagecodecs-lite tifffile scikit-image opencv-python numpy-stl \
    Pillow SimpleParse nibabel medpy SimpleITK mpi4py django rq mysqlclient
```

### Download or clone Biomedisa
```
# Clone repository
sudo apt-get install git
mkdir ~/git
cd ~/git
git clone https://github.com/biomedisa/biomedisa
```

### Adapt Biomedisa config
Make `config.py` as a copy of `config_example.py`
```
cp ~git/biomedisa/biomedisa_app/config_example.py ~git/biomedisa/biomedisa_app/config.py
```
In particular, adapt the following lines in `biomedisa/biomedisa_app/config.py`
```
'PATH_TO_BIOMEDISA' : '/home/dummy/git/biomedisa', # this is the path to your main biomedisa folder
'SECRET_KEY' : 'vl[cihu8uN!FrJoDbEqUymgMR()n}y7744$2;YLDm3Q8;MMX-g', # some random string
'DJANGO_DATABASE' : 'biomedisa_user_password', # password for the user 'biomedisa' of your biomedisa_database (set up in the next step)
'ALLOWED_HOSTS' : ['YOUR_IP', 'localhost', '0.0.0.0'], # you must tell django explicitly which hosts are allowed (e.g. your IP or the URL of your homepage)
'FIRST_QUEUE_NGPUS' : 4, # total number of GPUs available
```

### Install MySQL database
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

### Setting up CUDA environment

```
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update

# Install NVIDIA driver
sudo apt-get install --no-install-recommends nvidia-driver-450

# Reboot the system
sudo reboot

# Verify that NVIDIA driver can be loaded properly
nvidia-smi

# Install CUDA 10.1
sudo apt-get install --no-install-recommends cuda-10-1

# Add the following lines to your .bashrc (e.g. nano ~/.bashrc)
export CUDA_HOME=/usr/local/cuda-10.1
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

# Reload .bashrc and verify that CUDA is installed properly
source ~/.bashrc
nvcc --version

# Install PyCUDA
sudo -H PATH=/usr/local/cuda-10.1/bin:${PATH} pip3 install --upgrade pycuda

# Verify that PyCUDA is working properly
python3 ~/git/biomedisa/biomedisa_features/pycuda_test.py
```

# Install Tensorflow and Keras
```
# Install cuDNN
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update
sudo apt-get install --no-install-recommends \
    libcudnn7=7.6.5.32-1+cuda10.1 \
    libcudnn7-dev=7.6.5.32-1+cuda10.1

# Install Tensorflow and Keras
sudo -H pip3 install --upgrade tensorflow-gpu keras
```

# Run Biomedisa

Start workers (this has to be done after each reboot)
```
cd ~/git/biomedisa
./start_workers.sh
```

Start Biomedisa locally
```
python3 manage.py runserver 0.0.0.0:8000
```
# Open Biomedisa
Open Biomedisa in your local browser http://localhost:8000/ and log in as the `superuser` you created.

# Install Apache Server
Follow the [installation instructions](https://github.com/biomedisa/biomedisa/blob/master/README/INSTALL_APACHE_SERVER.md).
