import os
import subprocess

# base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# get biomedisa version
try:
    __version__ = subprocess.check_output(["git", "describe", "--tags", "--always"], cwd=BASE_DIR).decode('utf-8').strip()
    f = open(os.path.join(BASE_DIR,'log/biomedisa_version'), "w")
    f.write(__version__)
    f.close()
except:
    if os.path.isfile(os.path.join(BASE_DIR,'log/biomedisa_version')):
        __version__ = open(os.path.join(BASE_DIR,'log/biomedisa_version'), "r").readline().rstrip('\n')
    else:
        __version__ = 'not available'

