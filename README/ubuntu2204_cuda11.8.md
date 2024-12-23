#  Ubuntu 22.04 LTS + CUDA 11.8 (browser based)

- [Install Python and pip](#install-python-and-pip)
- [Install software dependencies](#install-software-dependencies)
- [Clone Biomedisa](#clone-biomedisa)
- [Install CUDA 11.8](#install-cuda-11.8)
- [Install cuDNN](#install-cudnn)
- [Install pip packages](#install-pip-packages)
- [Configure Biomedisa](#configure-biomedisa)
- [Install MySQL database](#install-mysql-database)
- [Run Biomedisa](#run-biomedisa)
- [Install Apache Server (optional)](#install-apache-server-optional)

#### Install Python and pip
```
sudo apt-get install python3 python3-dev python3-pip
```

#### Install software dependencies
```
sudo apt-get install libsm6 libxrender-dev libmysqlclient-dev pkg-config \
    libboost-python-dev build-essential screen libssl-dev cmake unzip \
    openmpi-bin openmpi-doc libopenmpi-dev redis-server git libgl1 wget
```

#### Clone Biomedisa
```
mkdir ~/git
cd ~/git
git clone https://github.com/biomedisa/biomedisa.git
```

#### Install CUDA 11.8
Biomedisa's Deep Learning framework requires TensorFlow 2.13, which is compatible with CUDA 11.8 and cuDNN 8.8.0. Please ensure that you install these specific versions, as higher versions are not yet supported. Add NVIDIA package repositories:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
```
If the error `W: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg)` occurs:
```
sudo apt-key export 3BF863CC | sudo gpg --dearmour -o /etc/apt/trusted.gpg.d/cuda-tools.gpg
```
Install CUDA Toolkit:
```
sudo apt-get update
sudo apt-get install --no-install-recommends cuda-11-8
```
Reboot and check that your GPUs are visible using the following command:
```
nvidia-smi
```

#### Adapt PATH variables
Add the Biomedisa base directory to the PYTHONPATH variable and the local pip directory to the PATH variable:
```
echo 'export PYTHONPATH=${HOME}/git/biomedisa:${PYTHONPATH}' >> ~/.bashrc
echo 'export PATH=${HOME}/.local/bin:${PATH}' >> ~/.bashrc
```
Add the CUDA paths:
```
echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64' >> ~/.bashrc
echo 'export PATH=${CUDA_HOME}/bin:${PATH}' >> ~/.bashrc
```
Reload .bashrc and verify that CUDA is installed properly:
```
source ~/.bashrc
nvcc --version
```

#### Install cuDNN
Install development and runtime libraries:
```
sudo apt-get install --no-install-recommends \
    libcudnn8=8.8.0.121-1+cuda11.8 \
    libcudnn8-dev=8.8.0.121-1+cuda11.8
```
OPTIONAL: hold packages to avoid crash after a system update:
```
sudo apt-mark hold libcudnn8 libcudnn8-dev cuda-11-8
```

#### Install pip packages
Download list of requirements and install pip packages. Note: If you run Biomedisa with an Apache Server (optional), you must install your packages system-wide using `sudo -H python3 -m pip install <package>`.
```
wget https://raw.githubusercontent.com/biomedisa/biomedisa/refs/heads/master/requirements.txt
python3 -m pip install -r requirements.txt
python3 -m pip install mysqlclient rq wget django==3.2.6
PATH=/usr/local/cuda-11.8/bin:${PATH} python3 -m pip install pycuda
```

#### Verify that PyCUDA is working properly
```
python3 -m biomedisa.features.pycuda_test
```

#### Verify that TensorFlow detects your GPUs
```
python3 -c "import tensorflow as tf; print('Detected GPUs:', len(tf.config.list_physical_devices('GPU')))"
```

#### Configure Biomedisa
Make `config.py` as a copy of `config_example.py`:
```
cp ~/git/biomedisa/biomedisa_app/config_example.py ~/git/biomedisa/biomedisa_app/config.py
```
Adapt the following lines in `~/git/biomedisa/biomedisa_app/config.py`:
```
'SECRET_KEY' : 'vl[cihu8uN!FrJoDbEqUymgMR()n}y7744$2;YLDm3Q8;MMX-g', # some random string
'DJANGO_DATABASE' : 'biomedisa_user_password', # password for the user 'biomedisa' that you created in the previous step
'ALLOWED_HOSTS' : ['localhost', '0.0.0.0'], # you must tell django explicitly which hosts are allowed (e.g. your IP and/or the URL of your homepage when running an APACHE server)
```

#### Install MySQL database
Install MySQL:
```
sudo apt-get install mysql-server
```
Login to MySQL (as root):
```
sudo mysql -u root -p
```
If the error `ERROR 2002 (HY000): Can't connect to local MySQL server through socket '/var/run/mysqld/mysqld.sock' (2)` occurs:
```
sudo service mysql stop
sudo usermod -d /var/lib/mysql/ mysql
sudo service mysql start
sudo mysql -u root -p
```
Set root password for MySQL database:
```
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'biomedisa_root_password';
quit;
sudo service mysql stop
sudo service mysql start
```
Login to MySQL (with the 'biomedisa_root_password' you just set):
```
mysql -u root -p
```
Create a user 'biomedisa' with password 'biomedisa_user_password' (same as set for 'DJANGO_DATABASE'):
```
CREATE USER 'biomedisa'@'localhost' IDENTIFIED BY 'biomedisa_user_password';
GRANT ALL PRIVILEGES ON *.* TO 'biomedisa'@'localhost' WITH GRANT OPTION;
```
Create Biomedisa database:
```
CREATE DATABASE biomedisa_database;
exit;
```
Add the following line to `/etc/mysql/mysql.conf.d/mysqld.cnf`:
```
wait_timeout = 604800
```
Migrate the database and create a superuser:
```
cd ~/git/biomedisa
python3 manage.py migrate
python3 manage.py createsuperuser
```

#### Run Biomedisa
Start workers (this has to be done after each reboot):
```
cd ~/git/biomedisa
./start_workers.sh
```

Start Biomedisa locally:
```
python3 manage.py runserver localhost:8080
```

#### Open Biomedisa
Open Biomedisa in your local browser http://localhost:8080/ and log in as the `superuser` you created.

#### Install Apache Server (optional)
Follow the [installation instructions](https://github.com/biomedisa/biomedisa/blob/master/README/APACHE_SERVER.md).

#### Update Biomedisa
Change to the Biomedisa directory and make a pull request:
```
cd ~/git/biomedisa
git pull
```

Update the database:
```
python3 manage.py migrate
```

If you installed an [Apache Server](https://github.com/biomedisa/biomedisa/blob/master/README/APACHE_SERVER.md), you need to restart the server:
```
sudo service apache2 restart
```

