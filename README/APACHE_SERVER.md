# Installation of Apache Server (optional)
Please note that both the Biomedisa repository and the Biomedisa Python environment are located under /opt/ using a user called 'dummy' who needs to run `./start_workers.sh`.

### Install Apache packages
```
sudo apt-get install apache2 apache2-doc libapache2-mod-wsgi-py3
```

### Edit Apache configeration
```
# Edit config file
sudo nano /etc/apache2/sites-available/000-default.conf

# Add the following lines between <VirtualHost *:80> and </VirtualHost>
# Replace <USER> properly

        Alias /static /opt/biomedisa/biomedisa_app/static
        <Directory /opt/biomedisa/biomedisa_app/static>
        Require all granted
        </Directory>

        Alias /media /opt/biomedisa/media
        <Directory /opt/biomedisa/media>
        Require all granted
        </Directory>

        Alias /paraview /opt/biomedisa/biomedisa_app/paraview
        <Directory /opt/biomedisa/biomedisa_app/paraview>
        Require all granted
        </Directory>

        <Directory /opt/biomedisa/biomedisa>
        <Files wsgi.py>
        Require all granted
        </Files>
        </Directory>

        WSGIScriptAlias / /opt/biomedisa/biomedisa/wsgi.py
        WSGIApplicationGroup %{GLOBAL}
        WSGIDaemonProcess biomedisa \
            python-path=/opt/biomedisa/biomedisa \
            python-home=/opt/biomedisa_env
        WSGIProcessGroup biomedisa
```

### Set permissions for www-data
```
sudo addgroup biomedisa
sudo usermod -aG biomedisa www-data
sudo usermod -aG biomedisa dummy

sudo chown -R dummy:biomedisa /opt/biomedisa_env
sudo chmod -R 755 /opt/biomedisa_env
sudo find /opt/biomedisa_env -type d -exec chmod g+s {} +

sudo chown -R dummy:biomedisa /opt/biomedisa
sudo chmod -R 750 /opt/biomedisa
sudo chmod -R 770 /opt/biomedisa/private_storage \
    /opt/biomedisa/media \
    /opt/biomedisa/log

sudo find /opt/biomedisa/private_storage -type d -exec chmod g+s {} +
sudo find /opt/biomedisa/media -type d -exec chmod g+s {} +
sudo find /opt/biomedisa/log -type d -exec chmod g+s {} +
```

### Restart Apache Server
```
sudo service apache2 restart
```
