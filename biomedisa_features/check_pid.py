import django
django.setup()
import os,sys
from biomedisa_app.models import Upload
from biomedisa_app.config import config
from biomedisa_app.views import send_error_message
import time

if __name__ == "__main__":
    try:
        os.kill(int(sys.argv[1]), 0)
    except OSError:
        image = Upload.objects.get(pk=sys.argv[2])
        if image.status == 2:
            image.status = 0
            image.pid = 0
            image.save()
            error_message = 'Something went wrong. Please restart.'
            Upload.objects.create(user=image.user, project=image.project, log=1, imageType=None, shortfilename=error_message)
            path_to_logfile = config['PATH_TO_BIOMEDISA'] + '/log/logfile.txt'
            with open(path_to_logfile, 'a') as logfile:
                print('%s %s %s %s' %(time.ctime(), image.user.username, image.shortfilename, error_message), file=logfile)
            send_error_message(image.user.username, image.shortfilename, error_message)
            print("pid is unassigned")
    else:
        print("pid is in use")
