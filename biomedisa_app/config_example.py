config = {
    'SERVER' : 'http://localhost:8080', # localhost, URL of your homepage e.g. 'https://biomedisa.info' or your internal IP e.g. 'http://192.168.176.30'
    'SERVER_ALIAS' : 'biomedisa-001', # an alias name for your server (for email notification and logfiles)
    'SECRET_KEY' : '...', # some random string
    'DJANGO_DATABASE' : '...', # password of your mysql database
    'ALLOWED_HOSTS' : ['localhost', '0.0.0.0'], # you must tell django explicitly which hosts are allowed (e.g. your IP and/or the URL of your homepage when running an APACHE server)
    'SECURE_MODE' : False, # this mode is highly recommended if you use biomedisa for production with users you do not trust
    'DEBUG' : True, # activate the debug mode if you develop the app. This must be deactivated in production mode for security reasons!

    'EMAIL_CONFIRMATION' : False, # users must confirm their emails during the registration process (to handle biomedisa's notification service the following email support must be set up)
    'EMAIL' : 'philipp.loesel@anu.edu.au',
    'EMAIL_USER' : '...',
    'EMAIL_PASSWORD' : '...',
    'SMTP_SEND_SERVER' : 'smtp.office365.com',
    'SMTP_PORT' : 587,

    'FIRST_QUEUE_HOST' : '', # empty string ('') if it is running on your local machine
    'FIRST_QUEUE_NGPUS' : 'all', # number of GPUs available (e.g. 1, 4, 'all') or list of GPU IDs (e.g. [0,3]). If CLUSTER=True this must be the sum of of all GPUs. List and 'all' works only locally and with CUDA.
    'FIRST_QUEUE_CLUSTER' : False, # if you want to use several machines for one queue (see README/INSTALL_CLUSTER.txt), you must specify the IPs of your machines and the number of GPUs respectively in 'log/workers_host'

    'SECOND_QUEUE' : False, # use an additional queue
    'SECOND_QUEUE_HOST' : 'dummy@192.168.176.31', # empty string ('') if it is running on your local machine
    'SECOND_QUEUE_NGPUS' : 4, # number of GPUs available (e.g. 1, 4, 'all') or list of GPU IDs (e.g. [0,3]). If CLUSTER=True this must be the sum of of all GPUs. List and 'all' works only locally and with CUDA.
    'SECOND_QUEUE_CLUSTER' : False, # if you want to use several machines for one queue (see README/INSTALL_CLUSTER.txt), you must specify the IPs of your machines and the number of GPUs respectively in 'log/workers_host'

    'THIRD_QUEUE' : False, # seperate queue for AI training. If False, AI tasks are queued in first queue
    'THIRD_QUEUE_HOST' : '', # empty string ('') if it is running on your local machine
    'THIRD_QUEUE_NGPUS' : 'all', # number of GPUs available (e.g. 1, 4, 'all') or list of GPU IDs (e.g. [0,3]). Works only locally.
    }
