# Installation of Apache Server (optional)

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

        Alias /static /home/<USER>/git/biomedisa/biomedisa_app/static
        <Directory /home/<USER>/git/biomedisa/biomedisa_app/static>
        Require all granted
        </Directory>

        Alias /media /home/<USER>/git/biomedisa/media
        <Directory /home/<USER>/git/biomedisa/media>
        Require all granted
        </Directory>

        Alias /paraview /home/<USER>/git/biomedisa/biomedisa_app/paraview
        <Directory /home/<USER>/git/biomedisa/biomedisa_app/paraview>
        Require all granted
        </Directory>

        <Directory /home/<USER>/git/biomedisa/biomedisa>
        <Files wsgi.py>
        Require all granted
        </Files>
        </Directory>

        WSGIScriptAlias / /home/<USER>/git/biomedisa/biomedisa/wsgi.py
        WSGIApplicationGroup %{GLOBAL}
        WSGIDaemonProcess biomedisa python-path=/home/<USER>/git/biomedisa/biomedisa
        WSGIProcessGroup biomedisa
```

### Set permissions for www-data
```
sudo usermod -aG www-data <USER>
sudo usermod -aG <USER> www-data

sudo chown -R <USER>:<USER> ~/git/biomedisa
chmod -R 750 ~/git/biomedisa
chmod -R 770 ~/git/biomedisa/private_storage
chmod -R 770 ~/git/biomedisa/media
chmod -R 770 ~/git/biomedisa/log
chmod -R 770 ~/git/biomedisa/tmp
```

### Restart Apache Server
```
sudo service apache2 restart
```
