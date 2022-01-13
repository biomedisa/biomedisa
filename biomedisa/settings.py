import os, sys

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from biomedisa_app.config import config

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = config['SECRET_KEY']

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = config['DEBUG']

ALLOWED_HOSTS = config['ALLOWED_HOSTS']

# Application definition

INSTALLED_APPS = [
    'biomedisa_app.apps.BiomedisaAppConfig',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'biomedisa.urls'
LOGIN_URL = '/login/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
MEDIA_URL = '/media/'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'biomedisa.wsgi.application'
FILE_UPLOAD_PERMISSIONS = 0o660
PRIVATE_STORAGE_ROOT = os.path.join(BASE_DIR, 'private_storage')
PRIVATE_STORAGE_URL = '/private_storage/'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'biomedisa_database',
        'USER': ('biomedisa' if config['OS'] == 'linux' else 'root'),
        'PASSWORD': config['DJANGO_DATABASE'],
        'OPTIONS': {
        'init_command': "SET sql_mode='STRICT_TRANS_TABLES'",
        }
    }
}
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'Europe/Berlin'

USE_I18N = True

USE_L10N = True

USE_TZ = True

# Paraview files
PARAVIEW_URL = '/paraview/'
PARAVIEW_ROOT = os.path.join(BASE_DIR, 'biomedisa_app/paraview/')

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'biomedisa_app/static')

# Email service
EMAIL_USE_TLS = True
EMAIL_HOST = config['SMTP_SEND_SERVER']
EMAIL_PORT = config['SMTP_PORT']
EMAIL_HOST_USER = config['EMAIL_USER']
EMAIL_HOST_PASSWORD = config['EMAIL_PASSWORD']
DEFAULT_FROM_EMAIL = 'Biomedisa <'+config['EMAIL']+'>'
