#!/bin/bash

##########################################################################
##                                                                      ##
##  Copyright (c) 2020 Philipp Lösel. All rights reserved.              ##
##                                                                      ##
##  This file is part of the open source project biomedisa.             ##
##                                                                      ##
##  Licensed under the European Union Public Licence (EUPL)             ##
##  v1.2, or - as soon as they will be approved by the                  ##
##  European Commission - subsequent versions of the EUPL;              ##
##                                                                      ##
##  You may redistribute it and/or modify it under the terms            ##
##  of the EUPL v1.2. You may not use this work except in               ##
##  compliance with this Licence.                                       ##
##                                                                      ##
##  You can obtain a copy of the Licence at:                            ##
##                                                                      ##
##  https://joinup.ec.europa.eu/page/eupl-text-11-12                    ##
##                                                                      ##
##  Unless required by applicable law or agreed to in                   ##
##  writing, software distributed under the Licence is                  ##
##  distributed on an "AS IS" basis, WITHOUT WARRANTIES                 ##
##  OR CONDITIONS OF ANY KIND, either express or implied.               ##
##                                                                      ##
##  See the Licence for the specific language governing                 ##
##  permissions and limitations under the Licence.                      ##
##                                                                      ##
##########################################################################

# bash security enhancements
set -eEuo pipefail
IFS=$'\n\t'

#####################################################################
#                                                                   #
# install script for the biomedisa server                           #
#                                                                   #
#####################################################################

#####################################################################
# Basic file system paths                                           #
#####################################################################

# home directory for the biomedisa user
BIOMEDISA_USER_HOME="/opt/biomedisa_home"
# install directory for biomedisa (should be owned by the biomedisa user
BIOMEDISA_INSTALL_DIR=${BIOMEDISA_USER_HOME}"/biomedisa"
# because the installation takes a lot of time this script calls sudo on a regular basis
# therefore a temporary file is needed
SUDO_STAT_FILE="/tmp/sudo_status.txt"

#####################################################################
# Lists of packages to install                                      #
#####################################################################

# packages installed using the apt package manager
APT_PACKAGE_LIST="python3
                  python3-dev
                  python3-pip
                  python3-testresources
                  python-gdcm
                  libsm6
                  libxrender-dev
                  libboost-python-dev
                  build-essential
                  redis-server
                  libssl-dev
                  cmake
                  libmysqlclient-dev
                  pwgen
                  openmpi-bin
                  openmpi-doc
                  libopenmpi-dev
                  ubuntu-drivers-common
                  acl
                  apache2
                  libapache2-mod-wsgi-py3
                  "
# packages installed using the pip3 package manager
PIP_PACKAGE_LIST="setuptools
                  scikit-build
                  scipy
                  scikit-image
                  opencv-python
                  h5py
                  pydicom
                  Pillow
                  tifffile
                  SimpleParse
                  SimpleITK
                  nibabel
                  medpy
                  numba
                  numpy
                  django
                  rq
                  mysqlclient
                  pip
                  enum34
                  futures
                  mpi4py
                  colorama
                  imagecodecs-lite
                  "

#####################################################################
# Lists of redis workers to create                                  #
#####################################################################

REDIS_WORKER="first_queue
              second_queue
              third_queue
              slices
              acwe
              cleanup
              share_notification
              stop_job
              load_data
              "

#####################################################################
# Compleatly remove a mysql installation                            #
#####################################################################
function purge_mysql {
    for p in $(dpkg -l | grep -E 'ii +mysql-' | awk '{print $2}')
    do
        sudo -H sh -c "DEBIAN_FRONTEND=noninteractive apt-get purge -y ${p}" || true
    done
    sudo rm -rf /etc/mysql /var/lib/mysql || true
}

#####################################################################
# Setups the mysql server for biomedisa (multipurpose               #
#       * asks for OR sets a mysql root passwd                      #
#       * asks for OR generates a mysql user passwd for biomedisa   #
#       * creates biomedisa database and sets permissions           #
#####################################################################
function setup_mysql_db {
    echo -e "\n\nInstalling mysql"
    if ! dpkg -l | grep -E "ii +mysql-server +" > /dev/null
    then
        # we need to install mysql-server
        while true; do
            echo "You have to set a mysql root password:"
            read -s -p "Password: " MYSQL_ROOT_PWD
            echo
            read -s -p "Password (again): " MYSQL_ROOT_PWD2
            echo
            if [[ "${MYSQL_ROOT_PWD}" = "${MYSQL_ROOT_PWD2}" ]] ; then
                break
            fi
            echo "Please try again"
        done
        
        sudo debconf-set-selections <<< "mysql-server mysql-server/root_password password ${MYSQL_ROOT_PWD}"
        sudo debconf-set-selections <<< "mysql-server mysql-server/root_password_again password ${MYSQL_ROOT_PWD}"
        sudo apt-get -y -q install mysql-server
    else
        # we to get the mysql root password
        while true; do
            echo -e "\nPlease enter the mysql root password:"
            echo -e "\tDANGER-ZONE: If you want to fully remove mysql and start over type \"ireallywanttoremovemysql\" in capital letters!"
            read -s -p "Password: " MYSQL_ROOT_PWD
            echo
            if [[ "${MYSQL_ROOT_PWD}" = "IREALLYWANTTOREMOVEMYSQL" ]] ; then
                purge_mysql
                ./$0
                exit 0
            fi
            if mysql -uroot -p${MYSQL_ROOT_PWD} -e "SELECT 1;" > /dev/null
            then
                break
            fi
            echo "Wrong passwd, please try again"
        done
    fi

    echo -e "\n\nConfiguring mysql"
    MYSQL_BIOMEDISA_PWD=$(pwgen -C -n -s -B 16 1)

    if mysql -uroot -p${MYSQL_ROOT_PWD} -e "CREATE USER 'biomedisa'@'localhost' IDENTIFIED WITH mysql_native_password BY '${MYSQL_BIOMEDISA_PWD}';"
    then
        echo -e "\tMYSQL user 'biomedisa'@'localhost' created"
    else
        echo -e "\nDatabase user 'biomedisa'@'localhost' already exists'!"
        while true; do
            echo -e "\nPlease enter the password of 'biomedisa'@'localhost'."
            echo -e "\tDANGER-ZONE: If you want to fully remove mysql and start over type \"ireallywanttoremovemysql\" in capital letters!"
            read -s -p "Password: " MYSQL_BIOMEDISA_PWD
            if [[ "${MYSQL_BIOMEDISA_PWD}" = "IREALLYWANTTOREMOVEMYSQL" ]] ; then
                purge_mysql
                ./$0
                exit 0
            fi
            if mysql -ubiomedisa -p${MYSQL_BIOMEDISA_PWD} -e "SELECT 1;" > /dev/null
            then
                break
            fi
        done
    fi
    mysql -uroot -p${MYSQL_ROOT_PWD} -e "CREATE DATABASE IF NOT EXISTS biomedisa_database;" > /dev/null
    mysql -uroot -p${MYSQL_ROOT_PWD} -e "GRANT ALL ON biomedisa_database.* TO 'biomedisa'@'localhost';" > /dev/null
    echo -e "\tMYSQL database 'biomedisa_database' created\n"
}

#####################################################################
# Installs packages using the apt package manager                   #
##################################################################### 
function install_apt_packages {
    echo -e "\n\nInstalling dependencies"
    sudo apt-get update
    for p in ${APT_PACKAGE_LIST} ; do
        p=${p}
        p="${p#"${p%%[![:space:]]*}"}"
        p="${p%"${p##*[![:space:]]}"}"
        sudo -H sh -c "DEBIAN_FRONTEND=noninteractive apt-get install -y -q ${p}"
    done
}

#####################################################################
# Installs packages using the pip3 package manager                  #
##################################################################### 
function install_pip_packages {
    echo -e "\n\nInstalling pip packages"
    sudo -H pip3 install --upgrade pip
    echo ${PIP_PACKAGE_LIST} | xargs sudo -H pip3 install --upgrade
}

#####################################################################
# Creates a UNIX user for biomedisa                                 #
##################################################################### 
function create_biomedisa_user {
    user_exists=$(cat /etc/passwd | awk -F: '{if ($1 == "biomedisa"){print $1;}}')
    if [[ -z ${user_exists} ]] ; then
        echo -e "\n\nCreating biomedisa user"
        sudo useradd --home-dir ${BIOMEDISA_USER_HOME} -M --shell /bin/false biomedisa
    fi
    sudo mkdir -p ${BIOMEDISA_USER_HOME}
    sudo chown biomedisa:biomedisa ${BIOMEDISA_USER_HOME}
    #sudo setfacl -dm o::- ${BIOMEDISA_USER_HOME}
}

#####################################################################
# Create shared folders allowing biomedisa to access uploaded data  #
##################################################################### 
function setup_shared_biomedisa_folder {
    if [[ $# -ne 1 ]] || [[ "$1" = "" ]] ; then
        echo "Wrong number of arguments supplied! $#"
        echo "This is a BUG!!"
        exit 127
    fi
    sudo mkdir -p "${BIOMEDISA_INSTALL_DIR}/$1"
    sudo chown -R biomedisa:biomedisa-web "${BIOMEDISA_INSTALL_DIR}/$1"
    sudo chmod 775 "${BIOMEDISA_INSTALL_DIR}/$1"
    sudo setfacl -dm g:biomedisa-web:rwX "${BIOMEDISA_INSTALL_DIR}/$1"
}

#####################################################################
# Creates biomedisa-web group and copies biomedisa to install dir   #
##################################################################### 
function copy_biomedisa_to_install_dir {
    echo -e "\n\nCopying biomedisa to ${BIOMEDISA_USER_HOME}"
    sudo cp -r $(/usr/bin/dirname -z $(readlink -f $0)) ${BIOMEDISA_USER_HOME}
    sudo chown -R biomedisa:biomedisa ${BIOMEDISA_USER_HOME}

    sudo groupadd biomedisa-web || true
    sudo usermod -aG biomedisa-web biomedisa
    sudo usermod -aG biomedisa-web www-data

    setup_shared_biomedisa_folder "private_storage"
    setup_shared_biomedisa_folder "media"
    setup_shared_biomedisa_folder "tmp"
    setup_shared_biomedisa_folder "log"

    sudo touch "${BIOMEDISA_INSTALL_DIR}/log/applications.txt"
    sudo touch "${BIOMEDISA_INSTALL_DIR}/log/logfile.txt"
    sudo chown -R biomedisa:biomedisa-web "${BIOMEDISA_INSTALL_DIR}/log"
}

#####################################################################
# Asks for sudo password and loops in background to preserve rights #
##################################################################### 
function get_sudo {
    if ! sudo -v
    then
        echo -e "This script needs sudo priveledges!\nPlease enter your sudo password:"
        exit 1  
    fi
    
    echo $$ >> ${SUDO_STAT_FILE}
    while [[ -f ${SUDO_STAT_FILE} ]]; do
        sudo -v
        sleep 5
    done &
}

#####################################################################
# Installs and setups CUDA                                          #
#       * downloads and installs CUDA                               #
#       * asks for a reboot                                         #
#       * checks installation success after reboot                  #
##################################################################### 
function install_cuda10 {
    if ! dpkg -l | grep -E "ii +cuda-10-1 +" > /dev/null
    then
        echo -e "\n\nInstalling CUDA 10.1"
        wget -O /tmp/cuda-repo-ubuntu1804.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb
        sudo dpkg -i /tmp/cuda-repo-ubuntu1804.deb
        sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
        sudo apt-get update
        sudo apt-get -f install -y
        sudo mkdir -p /usr/lib/nvidia
        sudo apt-get install --no-install-recommends -q -y nvidia-driver-450
        sudo apt-get install -y -q cuda-10-1
        echo "Please restart your computer and check the driver using"
        echo -e "\tnvidia-smi"
        echo "Then run this script again to complete the installation."
        exit 0
    else
        echo -e "\n\nTesting NVIDIA driver"
        if ! nvidia-smi
        then
            echo -e "The GPU driver is not correctly installed! Use\n\tubuntu-drivers devices\nto find and install the missing driver!"
            exit 1
        fi
    fi

}

#####################################################################
# Installs pycuda providing path to CUDA installation               #
#####################################################################
function install_pycuda {
    # fix from https://chainer.readthedocs.io/en/v1.2.0/tips.html
    echo -e "\n\nInstalling pycuda"
    sudo apt-get update
    sudo apt-get install python3-pip
    sudo -H sh -c "PATH=/usr/local/cuda/bin:\$PATH pip3 install --upgrade pycuda"
    echo -e "\nTesting pycuda installation:"
    #sudo -u biomedisa python3 ${BIOMEDISA_INSTALL_DIR}/biomedisa_features/pycuda_test.py
}

#####################################################################
# Installs tensorflow and karas checking the dependencies           #
#####################################################################
function install_nvidia_tensorflow {
    pkg_count=0
    if dpkg -l | grep -E "ii +libcudnn7 +" > /dev/null
    then
        pkg_count=$((${pkg_count}+1))
    fi
    if dpkg -l | grep -E "ii +libcudnn7-dev +" > /dev/null
    then
        pkg_count=$((${pkg_count}+1))
    fi
    if [[ ${pkg_count} -ne 2 ]] ; then
        echo -e "\n\nInstalling libcudnn7"
        wget -O /tmp/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
        sudo dpkg -i /tmp/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
        sudo apt-get update
        sudo apt-get -f install
        sudo apt-get install --no-install-recommends libcudnn7=7.6.5.32-1+cuda10.1 libcudnn7-dev=7.6.5.32-1+cuda10.1
        # pinn the libcudnn7 version
        # sudo -H sh -c "echo 'libcudnn7 hold' | dpkg --set-selections"
        # sudo -H sh -c "echo 'libcudnn7-dev hold' | dpkg --set-selections"
    fi
    echo -e "\n\nInstalling tensorflow and keras"
    sudo -H  pip3 install --upgrade tensorflow-gpu keras
}

#####################################################################
# Creates systemd service configurations for the redis workers      #
#####################################################################
function create_redis_service_file {
    if [[ $# -ne 1 ]] || [[ "$1" = "" ]] ; then
        echo "Wrong number of arguments supplied! $#"
        echo "This is a BUG!!"
        exit 127
    fi
    file="\n"
    file=${file}"[Unit]\n"
    file=${file}"Description=Biomedisa redis communication queue \"$1\"\n"
    file=${file}"Wants=redis.service\n"
    file=${file}"After=redis.service\n"
    file=${file}"\n\n"
    file=${file}"[Service]\n"
    file=${file}"ProtectSystem=strict\n"
    file=${file}"ProtectHome=yes\n"
    if [[ "$1" != "first_queue" ]] ; then
        file=${file}"PrivateDevices=yes\n"
    fi
    file=${file}"ProtectKernelModules=yes\n"
    file=${file}"ProtectKernelTunables=yes\n"
    file=${file}"LockPersonality=yes\n"
    file=${file}"ProtectControlGroups=yes\n"
    if [[ "$1" != "first_queue" ]] && [[ "$1" != "second_queue" ]] && [[ "$1" != "acwe" ]]  ; then
        file=${file}"MemoryDenyWriteExecute=yes\n"
    fi
    file=${file}"ReadWritePaths=${BIOMEDISA_USER_HOME}\n"
    file=${file}"PrivateMounts=yes\n"
    file=${file}"PrivateUsers=yes\n"
    file=${file}"PrivateTmp=yes\n"
    file=${file}"NoNewPrivileges=yes\n"
    file=${file}"SystemCallArchitectures=native\n"
    file=${file}"\n"
    file=${file}"User=biomedisa\n"
    file=${file}"Environment=\"CUDA_HOME=/usr/local/cuda\" \"LD_LIBRARY_PATH=/usr/local/cuda/lib64\" \"PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\" \"PYTHONPATH=${BIOMEDISA_INSTALL_DIR}\" \"DJANGO_SETTINGS_MODULE=biomedisa.settings\"\n"
    file=${file}"WorkingDirectory=${BIOMEDISA_INSTALL_DIR}\n"
    file=${file}"\n\n"
    file=${file}"Type=simple\n"
    file=${file}"ExecStart=/usr/local/bin/rq worker $1\n"
    file=${file}"\n"
    file=${file}"[Install]\n"
    file=${file}"RequiredBy=multi-user.target\n"
}

#####################################################################
# Creates and installs the systemd services for the redis workers   #
#####################################################################
function create_redis_systemd_services {
    echo -e "\n\nCreating systemd services"
    IFS=$' \n'
    for rq in ${REDIS_WORKER} ; do
        # trim whitespaces
	    rq="${rq#"${rq%%[![:space:]]*}"}"
        rq="${rq%"${rq##*[![:space:]]}"}"

        create_redis_service_file ${rq}
        echo -e ${file} | sudo tee /etc/systemd/system/biomedisa_rq_worker_${rq}.service > /dev/null
        echo "biomedisa_rq_worker_${rq}.service"
    done
    sudo systemctl daemon-reload
    for rq in ${REDIS_WORKER} ; do
        # trim whitespaces
	    rq="${rq#"${rq%%[![:space:]]*}"}"
        rq="${rq%"${rq##*[![:space:]]}"}"

        sudo systemctl enable biomedisa_rq_worker_${rq}.service
        sudo systemctl restart biomedisa_rq_worker_${rq}.service
    done
    IFS=$'\n\t'
}

#####################################################################
# Creates configuration for apache2                                 #
#####################################################################
function create_apache2_config {
    echo -e "\n\nConfiguring apache2"
    file="\n"
    file=${file}"<VirtualHost *:80>\n"
    file=${file}"\n"
    file=${file}"\tServerAdmin webmaster@localhost\n"
    file=${file}"\tDocumentRoot /var/www/html\n"
    file=${file}"\tErrorLog \${APACHE_LOG_DIR}/error.log\n"
    file=${file}"\tCustomLog \${APACHE_LOG_DIR}/access.log combined\n"
    file=${file}"\n"
    file=${file}"\tAlias /static ${BIOMEDISA_INSTALL_DIR}/biomedisa_app/static\n"
    file=${file}"\t<Directory ${BIOMEDISA_INSTALL_DIR}/biomedisa_app/static>\n"
    file=${file}"\tRequire all granted\n"
    file=${file}"\t</Directory>\n"
    file=${file}"\n"
    file=${file}"\tAlias /media ${BIOMEDISA_INSTALL_DIR}/media\n"
    file=${file}"\t<Directory ${BIOMEDISA_INSTALL_DIR}/media>\n"
    file=${file}"\tRequire all granted\n"
    file=${file}"\t</Directory>\n"
    file=${file}"\n"
    file=${file}"\t<Directory ${BIOMEDISA_INSTALL_DIR}/biomedisa>\n"
    file=${file}"\t<Files wsgi.py>\n"
    file=${file}"\tRequire all granted\n"
    file=${file}"\t</Files>\n"
    file=${file}"\t</Directory>\n"
    file=${file}"\n"
    file=${file}"\tWSGIScriptAlias / ${BIOMEDISA_INSTALL_DIR}/biomedisa/wsgi.py\n"
    file=${file}"\tWSGIApplicationGroup %{GLOBAL}\n"
    file=${file}"\tWSGIDaemonProcess biomedisa python-path=${BIOMEDISA_INSTALL_DIR}\n"
    file=${file}"\tWSGIProcessGroup biomedisa\n"
    file=${file}"\n"
    file=${file}"</VirtualHost>\n"
}

#####################################################################
# Setup of the apache2 webserver for biomedisa                      #
#####################################################################
function configure_apache2 {
    create_apache2_config
    echo -e ${file} | sudo tee /etc/apache2/sites-available/biomedisa.conf > /dev/null
    sudo rm /etc/apache2/sites-enabled/*
    sudo ln -s /etc/apache2/sites-available/biomedisa.conf /etc/apache2/sites-enabled/

    sudo a2enmod wsgi

    sudo systemctl enable apache2.service
    sudo systemctl restart apache2.service
}

#####################################################################
# Asks user to provide neccesary information for biomedisa config   #
#####################################################################
function create_biomedisa_config {
    echo -e "\n\nBiomedisa configuration:\n"
    echo "The URL of your homepage e.g. 'biomedisa.de' or your internal IP e.g. '192.168.176.33'"
    read -p "Server IP: " server_ip
    echo
    echo "An alias name for your server (for email notification and logfiles)"
    read -p "Server alias: " server_alias
    echo
    echo "Users must confirm their emails during the registration process (to handle biomedisa's notification service the following email support must be set up)"
    read -p "Use email confirmation:(y|N) " answer
    echo
    if [[ ${answer} =~ ^y(es)?$ ]] ; then
        email_confirmation="True"
        read -p "email address: " email_address
        read -p "email username: " email_user
        read -p "email password: " email_password
        read -p "email SMTP server: " email_smtp_server
        read -p "email SMTP port: " email_smtp_port
    else
        email_confirmation="False"
        email_address="user@example.com"
        email_user="username_dummy"
        email_password="password_dummy"
        email_smtp_server="smtp.example.com"
        email_smtp_port="587"
    fi
    BIOMEDISA_SECRET_KEY=$(pwgen -C -n -s -B 50 1)
    echo -e "\nYou must tell django explicitly which hosts are allowed (e.g. your IP or the URL of your homepage)"
    echo -e "Allowed host ip addresses: \tOne per line, at least your hostname/local IP, 0.0.0.0 for all other hosts."
    allowed_hosts=""
    while : ; do
        read -p "(end with empty line) " answer
        if [[ -z ${answer} ]] ; then
            if [[ -n ${allowed_hosts} ]] ; then
                break
            fi
            echo "At least one is reqired!"
        else
            if [[ -z ${allowed_hosts} ]] ; then
                allowed_hosts="'${answer}'"
            else
                allowed_hosts="${allowed_hosts}, '${answer}'"
            fi
        fi
    done
    allowed_hosts="[${allowed_hosts}]"
    echo
    read -p "Number of GPUs to available: " number_of_gpus

    file="\n"
    file=${file}"config = {\n"
    file=${file}"\t'OS' : 'linux', # either 'linux' or 'windows'\n"
    file=${file}"\t'SERVER' : '${server_ip}', # the URL of your homepage e.g. 'biomedisa.de' or your internal IP e.g. '192.168.176.33'\n"
    file=${file}"\t'SERVER_ALIAS' : '${server_alias}', # an alias name for your server (for email notification and logfiles)\n"
    file=${file}"\t'PATH_TO_BIOMEDISA' : '${BIOMEDISA_INSTALL_DIR}', # this is the path to your main biomedisa folder e.g. '/home/dummy/git/biomedisa'\n"
    file=${file}"\t'SECURE_MODE' : False, # supported only on linux (this mode is highly recommended if you use biomedisa for production with users you do not trust) \n"
    file=${file}"\n"
    file=${file}"\t'EMAIL_CONFIRMATION' : ${email_confirmation}, # users must confirm their emails during the registration process (to handle biomedisa's notification service the following email support must be set up)\n"
    file=${file}"\t'EMAIL' : '${email_address}',\n"
    file=${file}"\t'EMAIL_USER' : '${email_user}',\n"
    file=${file}"\t'EMAIL_PASSWORD' : '${email_password}',\n"
    file=${file}"\t'SMTP_SEND_SERVER' : '${email_smtp_server}',\n"
    file=${file}"\t'SMTP_PORT' : ${email_smtp_port},\n"
    file=${file}"\n"
    file=${file}"\t'SECRET_KEY' : '${BIOMEDISA_SECRET_KEY}',\n"
    file=${file}"\t'DJANGO_DATABASE' : '${MYSQL_BIOMEDISA_PWD}', # password of your mysql database\n"
    file=${file}"\t'ALLOWED_HOSTS' : ${allowed_hosts}, # you must tell django explicitly which hosts are allowed (e.g. your IP or the url of your homepage)\n"
    file=${file}"\t'DEBUG' : False, # activate the debug mode if you develop the app. This must be deactivated in production mode for security reasons!\n"
    file=${file}"\n"
    file=${file}"\t'FIRST_QUEUE_HOST' : '', # empty string ('') if it is running on your local machine\n"
    file=${file}"\t'FIRST_QUEUE_NGPUS' : ${number_of_gpus}, # total number of GPUs available. If FIRST_QUEUE_CLUSTER=True this must be the sum of of all GPUs\n"
    file=${file}"\t'FIRST_QUEUE_CLUSTER' : False, # if you want to use several machines for one queue (see README/INSTALL_CLUSTER.txt), you must specify the IPs of your machines and the number of GPUs respectively in 'log/workers_host'\n"
    file=${file}"\n"
    file=${file}"\t'SECOND_QUEUE' : False, # use an additional queue\n"
    file=${file}"\t'SECOND_QUEUE_HOST' : 'dummy@192.168.176.31', # empty string ('') if it is running on your local machine\n"
    file=${file}"\t'SECOND_QUEUE_NGPUS' : 4, # total number of GPUs available. If SECOND_QUEUE_CLUSTER=True this must be the sum of of all GPUs\n"
    file=${file}"\t'SECOND_QUEUE_CLUSTER' : False, # if you want to use several machines for one queue (see README/INSTALL_CLUSTER.txt), you must specify the IPs of your machines and the number of GPUs respectively in 'log/workers_host'\n"
    file=${file}"\n"
    file=${file}"\t'THIRD_QUEUE' : False, # seperate queue for AI. If False, AI tasks are queued in first queue\n"
    file=${file}"\t'THIRD_QUEUE_HOST' : '', # empty string ('') if it is running on your local machine\n"
    file=${file}"}\n"

    echo -e "\n\nBiomedisa website configuration"
    read -p "Admin username: " BIOMEDISA_WEB_ADMIN_NAME
    read -p "Admin email: " BIOMEDISA_WEB_ADMIN_EMAIL

    while true
    do
        echo "You have to set a web admin password:"
        read -s -p "Admin password: " BIOMEDISA_WEB_ADMIN_PWD
        echo
        read -s -p "Admin password (again): " BIOMEDISA_WEB_ADMIN_PWD2
        echo
        if [[ "${BIOMEDISA_WEB_ADMIN_PWD}" = "${BIOMEDISA_WEB_ADMIN_PWD2}" ]] ; then
            break
        fi
        echo "Please try again"
    done

}

#####################################################################
# Configurates and finishes biomedisa installation and setup        #
#####################################################################
function configure_biomedisa {
    where_am_i=$(pwd)
    create_biomedisa_config
    cd ${BIOMEDISA_INSTALL_DIR}
    echo -e "\n\nWriting biomedisa Configuration"
    echo -e ${file} | sudo -u biomedisa tee ${BIOMEDISA_INSTALL_DIR}/biomedisa_app/config.py > /dev/null
    echo -e "\nExecuting manage.py makemigrations"
    sudo -u biomedisa python3 ${BIOMEDISA_INSTALL_DIR}/manage.py makemigrations
    echo -e "\nExecuting manage.py migrate"
    sudo -u biomedisa python3 manage.py migrate
    #echo -e "\nExecuting manage.py createsuperuser"
    #sudo -u biomedisa python3 manage.py createsuperuser
    # replaced due to this post https://stackoverflow.com/questions/6244382/how-to-automate-createsuperuser-on-django
    echo -e "\nExecuting manage.py createsuperuser"
    cat << EOF | sudo -u biomedisa python3 ${BIOMEDISA_INSTALL_DIR}/manage.py shell
from django.contrib.auth import get_user_model
User = get_user_model()
User.objects.create_superuser('${BIOMEDISA_WEB_ADMIN_NAME}','${BIOMEDISA_WEB_ADMIN_EMAIL}','${BIOMEDISA_WEB_ADMIN_PWD}')
EOF
    cd ${where_am_i}
}

#####################################################################
# After finishing the installation print the generated keys         #
#####################################################################
function print_keys {
    echo -e "\n\nPasswords:\n"
    set +u
    if [[ -n ${MYSQL_BIOMEDISA_PWD+x} ]]; then echo "mysql: biomedisa password:         ${MYSQL_BIOMEDISA_PWD}" ; fi
    if [[ -n ${MYSQL_ROOT_PWD+x} ]]; then echo "mysql: root password:              ${MYSQL_ROOT_PWD}" ; fi
    echo
    if [[ -n ${BIOMEDISA_SECRET_KEY+x} ]]; then echo "biomedisa: django secret key:      ${BIOMEDISA_SECRET_KEY}" ; fi
    echo
    if [[ -n ${BIOMEDISA_WEB_ADMIN_NAME+x} ]]; then echo "biomedisa: django admin username:  ${BIOMEDISA_WEB_ADMIN_NAME}" ; fi
    if [[ -n ${BIOMEDISA_WEB_ADMIN_EMAIL+x} ]]; then echo "biomedisa: django admin email:     ${BIOMEDISA_WEB_ADMIN_EMAIL}" ; fi
    if [[ -n ${BIOMEDISA_WEB_ADMIN_PWD+x} ]]; then echo "biomedisa: django admin password:  ${BIOMEDISA_WEB_ADMIN_PWD}" ; fi
    set -u
}

#####################################################################
# Function to catch errors during install (prints traceback)        #
#####################################################################
function onerr {
    echo -e "\n\nSomething went wrong!"
    echo "Error in ${BASH_COMMAND:-unknown}"
    traceback 1
    exit 1
}

#####################################################################
# Print traceback in case of error                                  #
#####################################################################
function traceback
{
  # Hide the traceback() call.
  local -i start=$(( ${1:-0} + 1 ))
  local -i end=${#BASH_SOURCE[@]}
  local -i i=0
  local -i j=0

  echo "Traceback (last called is first):" 1>&2
  for ((i=start; i < end; i++)); do
    j=$(( i - 1 ))
    local function="${FUNCNAME[$i]}"
    local file="${BASH_SOURCE[$i]}"
    local line="${BASH_LINENO[$j]}"
    echo "     ${function}() in ${file}:${line}" 1>&2
  done
}

#####################################################################
# Main function for the biomedisa installation                      #
#####################################################################
function install {
    get_sudo

    # This needs a reboot after installing the nvidia driver
    install_cuda10

    install_apt_packages
    install_pip_packages

    create_biomedisa_user
    copy_biomedisa_to_install_dir
    
    install_pycuda
    install_nvidia_tensorflow

    setup_mysql_db

    trap print_keys EXIT

    configure_biomedisa

    create_redis_systemd_services

    configure_apache2

    clear

    cat << EOF


  ##########################################################################
  ##                                                                      ##
  ##                  Successfully installed biomedisa!                   ##
  ##                                                                      ##
  ##                                                                      ##
  ##    If you plan to run biomedisa on a public server we                ##
  ##    STRONGLY recommend to enable SECURE_MODE!                         ##
  ##                                                                      ##
  ##    Edit "/opt/biomedisa_home/biomedisa_app/config.py" to do so.      ##
  ##                                                                      ##
  ##########################################################################

EOF

    # finish sudo loop
    rm ${SUDO_STAT_FILE}
}

#####################################################################
# Entry point of this install script                                #
#####################################################################

# setup the error trap
trap onerr ERR

# call to the main install function
install

