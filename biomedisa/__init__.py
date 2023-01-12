import os,subprocess
# Application version
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    __version__ = subprocess.check_output(["git", "describe", "--tags", "--always"], cwd=BASE_DIR).decode('utf-8').strip()
except:
    __version__ = '23.01.1'
