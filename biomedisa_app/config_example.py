config = {
    'OS' : 'linux', # either 'linux' or 'windows'
    'SERVER' : 'http://localhost:8080', # localhost, URL of your homepage e.g. 'https://biomedisa.org' or your internal IP e.g. 'http://192.168.176.30'
    'SERVER_ALIAS' : 'biomedisa-001', # an alias name for your server (for email notification and logfiles)
    'PATH_TO_BIOMEDISA' : '/home/dummy/git/biomedisa', # this is the path to your main biomedisa folder e.g. '/home/dummy/git/biomedisa'
    'SECRET_KEY' : '...', # some random string
    'DJANGO_DATABASE' : '...', # password of your mysql database
    'ALLOWED_HOSTS' : ['192.168.176.30', 'localhost', '0.0.0.0'], # you must tell django explicitly which hosts are allowed (e.g. your IP or the url of your homepage)
    'SECURE_MODE' : False, # supported only on linux (this mode is highly recommended if you use biomedisa for production with users you do not trust)
    'DEBUG' : False, # activate the debug mode if you develop the app. This must be deactivated in production mode for security reasons!

    'EMAIL_CONFIRMATION' : False, # users must confirm their emails during the registration process (to handle biomedisa's notification service the following email support must be set up)
    'EMAIL' : 'philipp.loesel@uni-heidelberg.de',
    'EMAIL_USER' : '...',
    'EMAIL_PASSWORD' : '...',
    'SMTP_SEND_SERVER' : 'mail.urz.uni-heidelberg.de',
    'SMTP_PORT' : 587,

    'FIRST_QUEUE_HOST' : '', # empty string ('') if it is running on your local machine
    'FIRST_QUEUE_NGPUS' : 4, # total number of GPUs available. If FIRST_QUEUE_CLUSTER=True this must be the sum of of all GPUs
    'FIRST_QUEUE_CLUSTER' : False, # if you want to use several machines for one queue (see README/INSTALL_CLUSTER.txt), you must specify the IPs of your machines and the number of GPUs respectively in 'log/workers_host'

    'SECOND_QUEUE' : False, # use an additional queue
    'SECOND_QUEUE_HOST' : 'dummy@192.168.176.31', # empty string ('') if it is running on your local machine
    'SECOND_QUEUE_NGPUS' : 4, # total number of GPUs available. If SECOND_QUEUE_CLUSTER=True this must be the sum of of all GPUs
    'SECOND_QUEUE_CLUSTER' : False, # if you want to use several machines for one queue (see README/INSTALL_CLUSTER.txt), you must specify the IPs of your machines and the number of GPUs respectively in 'log/workers_host'

    'THIRD_QUEUE' : False, # seperate queue for AI. If False, AI tasks are queued in first queue
    'THIRD_QUEUE_HOST' : '' # empty string ('') if it is running on your local machine
    }
