#  Ubuntu 20.04 LTS + CUDA 11.3 (full installation)

- [Install Python and pip](#install-python-and-pip)
- [Install software dependencies](#install-software-dependencies)
- [Install pip packages](#install-pip-packages)
- [Clone Biomedisa](#clone-biomedisa)
- [Install MySQL database](#install-mysql-database)
- [Install CUDA 11.3](#install-cuda-11.3)
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
    openmpi-bin openmpi-doc libopenmpi-dev redis-server git
```

#### Install pip packages
```
sudo -H pip3 install --upgrade pip setuptools testresources scikit-build
sudo -H pip3 install --upgrade numpy scipy h5py colorama wget numpy-stl \
    numba imagecodecs-lite tifffile scikit-image opencv-python \
    Pillow nibabel medpy SimpleITK mpi4py itk vtk rq mysqlclient matplotlib
sudo -H pip3 install django==3.2.6
```

#### Clone Biomedisa
```
mkdir ~/git
cd ~/git
git clone https://github.com/biomedisa/biomedisa.git
```

#### Adapt Biomedisa config
Make `config.py` as a copy of `config_example.py`
```
cp biomedisa/biomedisa_app/config_example.py biomedisa/biomedisa_app/config.py
```
In particular, adapt the following lines in `biomedisa/biomedisa_app/config.py`
```
'PATH_TO_BIOMEDISA' : '/home/dummy/git/biomedisa', # this is the path to your main biomedisa folder
'SECRET_KEY' : 'vl[cihu8uN!FrJoDbEqUymgMR()n}y7744$2;YLDm3Q8;MMX-g', # some random string
'DJANGO_DATABASE' : 'biomedisa_user_password', # password for the user 'biomedisa' of your biomedisa_database (set up in the next step)
'ALLOWED_HOSTS' : ['YOUR_IP', 'localhost', '0.0.0.0'], # you must tell django explicitly which hosts are allowed (e.g. your IP or the URL of your homepage)
```

#### Install MySQL database
```
# Install MySQL
sudo apt-get install mysql-server

# Login to MySQL (as root)
sudo mysql -u root -p

# If ERROR 2002 (HY000): Can't connect to local MySQL server through socket '/var/run/mysqld/mysqld.sock' (2)
sudo service mysql stop
sudo usermod -d /var/lib/mysql/ mysql
sudo service mysql start
sudo mysql -u root -p

# Set root password for MySQL database
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'biomedisa_root_password';
quit;
sudo service mysql stop
sudo service mysql start

# Login to MySQL (with the 'biomedisa_root_password' you just set)
mysql -u root -p

# Create a user 'biomedisa' with password 'biomedisa_user_password' (same as set for 'DJANGO_DATABASE')
CREATE USER 'biomedisa'@'localhost' IDENTIFIED BY 'biomedisa_user_password';
GRANT ALL PRIVILEGES ON *.* TO 'biomedisa'@'localhost' WITH GRANT OPTION;

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

#### Install CUDA 11.3
```
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install --no-install-recommends cuda-11-3

# Reboot. Check that GPUs are visible using the command
nvidia-smi

# Add the following lines to your .bashrc (e.g. nano ~/.bashrc)
export CUDA_HOME=/usr/local/cuda-11.3
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

# Reload .bashrc and verify that CUDA is installed properly
source ~/.bashrc
nvcc --version

# Install PyCUDA
sudo -H "PATH=/usr/local/cuda-11.3/bin:${PATH}" pip3 install --upgrade pycuda

# Verify that PyCUDA is working properly
python3 ~/git/biomedisa/biomedisa_features/pycuda_test.py
```

#### Install TensorFlow
```
wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
sudo apt-get update

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libnvinfer8_8.0.3-1+cuda11.3_amd64.deb
sudo apt install ./libnvinfer8_8.0.3-1+cuda11.3_amd64.deb
sudo apt-get update

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    libcudnn8=8.2.1.32-1+cuda11.3  \
    libcudnn8-dev=8.2.1.32-1+cuda11.3

# Install TensorRT. Requires that libcudnn8 is installed above.
sudo apt-get install --no-install-recommends libnvinfer8=8.0.3-1+cuda11.3 \
    libnvinfer-dev=8.0.3-1+cuda11.3 \
    libnvinfer-plugin8=8.0.3-1+cuda11.3

# OPTIONAL: hold packages to avoid crash after system update
sudo apt-mark hold libcudnn8 libcudnn8-dev libnvinfer-dev libnvinfer-plugin8 libnvinfer8 cuda-11-3

# Install TensorFlow
sudo -H pip3 install --upgrade tensorflow-gpu
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
