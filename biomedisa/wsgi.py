import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

os.environ["DJANGO_SETTINGS_MODULE"] = "biomedisa.settings"
os.environ.setdefault("LC_ALL", "en_US.UTF-8")
os.environ.setdefault("LC_CTYPE", "en_US.UTF-8")

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
