#  Ubuntu 22.04 LTS + CUDA 11.8 (full installation)

- [Install Python and pip](#install-python-and-pip)
- [Install software dependencies](#install-software-dependencies)
- [Install pip packages](#install-pip-packages)
- [Clone Biomedisa](#clone-biomedisa)
- [Install MySQL database](#install-mysql-database)
- [Install CUDA 11.8](#install-cuda-11.8)
- [Install TensorFlow](#install-tensorflow)
- [Run Biomedisa](#run-biomedisa)
- [Install Apache Server (optional)](#install-apache-server-optional)

#### Install Python and pip
```
sudo apt-get install python3 python3-dev python3-pip
```

#### Install software dependencies
```
sudo apt-get install libsm6 libxrender-dev libmysqlclient-dev pkg-config \
    libboost-python-dev build-essential screen libssl-dev cmake \
    openmpi-bin openmpi-doc libopenmpi-dev redis-server git libgl1
```

#### Install pip packages
Note: If you are not running Biomedisa with an Apache Server, you may only use `pip3 install --upgrade <package>` and add `export PATH=/home/$USER/.local/bin:${PATH}` to `~/.bashrc`.
```
sudo -H pip3 install --upgrade pip setuptools testresources scikit-build
sudo -H pip3 install --upgrade numpy scipy h5py colorama wget numpy-stl \
    numba imagecodecs tifffile scikit-image opencv-python \
    Pillow nibabel medpy SimpleITK mpi4py itk vtk rq mysqlclient matplotlib
sudo -H pip3 install django==3.2.6
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
'SECRET_KEY' : 'vl[cihu8uN!FrJoDbEqUymgMR()n}y7744$2;YLDm3Q8;MMX-g', # some random string
'DJANGO_DATABASE' : 'biomedisa_user_password', # password for the user 'biomedisa' that you created in the previous step
'ALLOWED_HOSTS' : ['localhost', '0.0.0.0'], # you must tell django explicitly which hosts are allowed (e.g. your IP and/or the URL of your homepage when running an APACHE server)
```

#### Install CUDA 11.8
```
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"

# If W: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg)
sudo apt-key export 3BF863CC | sudo gpg --dearmour -o /etc/apt/trusted.gpg.d/cuda-tools.gpg

# Install CUDA
sudo apt-get update
sudo apt-get install --no-install-recommends cuda-11-8

# Reboot. Check that GPUs are visible using the command
nvidia-smi

# Add the following lines to `~/.bashrc` (e.g. nano ~/.bashrc)
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

# Reload .bashrc and verify that CUDA is installed properly
source ~/.bashrc
nvcc --version

# Install PyCUDA
sudo -H "PATH=/usr/local/cuda-11.8/bin:${PATH}" pip3 install --upgrade pycuda

# Verify that PyCUDA is working properly
python3 ~/git/biomedisa/biomedisa_features/pycuda_test.py
```

#### Install TensorFlow
```
# Install development and runtime libraries.
sudo apt-get install --no-install-recommends \
    libcudnn8=8.8.0.121-1+cuda11.8 \
    libcudnn8-dev=8.8.0.121-1+cuda11.8

# Install TensorRT. Requires that libcudnn8 is installed above.
sudo apt-get install --no-install-recommends libnvinfer8=8.5.3-1+cuda11.8 \
    libnvinfer-dev=8.5.3-1+cuda11.8 \
    libnvinfer-plugin8=8.5.3-1+cuda11.8

# OPTIONAL: hold packages to avoid crash after system update
sudo apt-mark hold libcudnn8 libcudnn8-dev libnvinfer-dev libnvinfer-plugin8 libnvinfer8 cuda-11-8

# Install TensorFlow
sudo -H pip3 install tensorflow==2.13.0
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

