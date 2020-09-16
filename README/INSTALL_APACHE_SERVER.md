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
# Replace USER properly

        Alias /static /home/USER/git/biomedisa/biomedisa_app/static
        <Directory /home/USER/git/biomedisa/biomedisa_app/static>
        Require all granted
        </Directory>

        Alias /media /home/USER/git/biomedisa/media
        <Directory /home/USER/git/biomedisa/media>
        Require all granted
        </Directory>

        <Directory /home/USER/git/biomedisa/biomedisa>
        <Files wsgi.py>
        Require all granted
        </Files>
        </Directory>

        WSGIScriptAlias / /home/USER/git/biomedisa/biomedisa/wsgi.py
        WSGIApplicationGroup %{GLOBAL}
        WSGIDaemonProcess biomedisa python-path=/home/USER/git/biomedisa/biomedisa
        WSGIProcessGroup biomedisa
```

### Restart Apache Server
```
sudo service apache2 restart
```

### Add a new group `biomedisa-web` and set permissions properly
```
sudo groupadd biomedisa-web
sudo usermod -aG www-data $USER
sudo usermod -aG biomedisa-web $USER
sudo usermod -aG biomedisa-web www-data

sudo chown -R $USER:biomedisa-web ~/git/biomedisa/private_storage
sudo chown -R $USER:biomedisa-web ~/git/biomedisa/media
sudo chown -R $USER:biomedisa-web ~/git/biomedisa/log
sudo chown -R $USER:biomedisa-web ~/git/biomedisa/tmp
sudo chown -R $USER:biomedisa-web ~/git/biomedisa/biomedisa_app/config.py

chmod -R 770 ~/git/biomedisa/private_storage
chmod -R 770 ~/git/biomedisa/media
chmod -R 770 ~/git/biomedisa/log
chmod -R 770 ~/git/biomedisa/tmp
```