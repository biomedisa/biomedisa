import os,sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import biomedisa

def selected_settings(request):
    # return the version value as a dictionary
    # you may add other values here as well
    return {'APP_VERSION_NUMBER': biomedisa.__version__}
