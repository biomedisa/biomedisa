#  Ubuntu 22/24 LTS (browser based)

- [Install Python 3.10 and Pip](#install-python-and-pip)
- [Install Software Dependencies](#install-software-dependencies)
- [Clone Biomedisa](#clone-biomedisa)
- [Install CUDA Toolkit](#install-cuda-toolkit)
- [Install Pip Packages](#install-pip-packages)
- [Configure Biomedisa](#configure-biomedisa)
- [Install MySQL Database](#install-mysql-database)
- [Run Biomedisa](#run-biomedisa)
- [Install Apache Server (optional)](#install-apache-server-optional)

#### Install Python 3.10 and Pip
Ubuntu 22.04:
```
sudo apt-get install python3 python3-dev python3-pip
```
Ubuntu 24.04:
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.10 python3.10-dev python3-pip
```

#### Install Software Dependencies
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

#### Install CUDA Toolkit
Download and install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) or via command-line as follows. You may choose any CUDA version compatible with your NVIDIA GPU architecture as outlined in the [NVIDIA Documentation](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html). If you select a version other than CUDA 12.6 for Ubuntu 22.04, you will need to adjust the following steps accordingly:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```
Reboot and check that your GPUs are visible using the following command:
```
nvidia-smi
```
Add the CUDA paths (adjust the CUDA version if required):
```
echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64' >> ~/.bashrc
echo 'export PATH=${CUDA_HOME}/bin:${PATH}' >> ~/.bashrc
```
Reload .bashrc and verify that CUDA is installed properly:
```
source ~/.bashrc
nvcc --version
```

#### Install Pip Packages
Add the local pip directory to the PATH variable:
```
echo 'export PATH=${HOME}/.local/bin:${PATH}' >> ~/.bashrc
```
Install pip packages. Note: If you run Biomedisa with an Apache Server (optional), you must install your packages system-wide using `sudo -H python3 -m pip install ...`.
```
cd ~/git/biomedisa
python3 -m pip install -r requirements.txt
python3 -m pip install mysqlclient rq wget django==3.2.6
PATH=/usr/local/cuda-12.6/bin:${PATH} python3 -m pip install pycuda
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

