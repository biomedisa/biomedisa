##########################################################################
##                                                                      ##
##  Copyright (c) 2019-2024 Philipp Lösel. All rights reserved.         ##
##                                                                      ##
##  This file is part of the open source project biomedisa.             ##
##                                                                      ##
##  Licensed under the European Union Public Licence (EUPL)             ##
##  v1.2, or - as soon as they will be approved by the                  ##
##  European Commission - subsequent versions of the EUPL;              ##
##                                                                      ##
##  You may redistribute it and/or modify it under the terms            ##
##  of the EUPL v1.2. You may not use this work except in               ##
##  compliance with this Licence.                                       ##
##                                                                      ##
##  You can obtain a copy of the Licence at:                            ##
##                                                                      ##
##  https://joinup.ec.europa.eu/page/eupl-text-11-12                    ##
##                                                                      ##
##  Unless required by applicable law or agreed to in                   ##
##  writing, software distributed under the Licence is                  ##
##  distributed on an "AS IS" basis, WITHOUT WARRANTIES                 ##
##  OR CONDITIONS OF ANY KIND, either express or implied.               ##
##                                                                      ##
##  See the Licence for the specific language governing                 ##
##  permissions and limitations under the Licence.                      ##
##                                                                      ##
##########################################################################

import os
import biomedisa

def create_error_object(message, remote=False, queue=None, img_id=None):

    # remote server
    if remote:
        with open(biomedisa.BASE_DIR + f'/log/error_{queue}', 'w') as errorfile:
            print(message, file=errorfile)

    # local server
    else:
        import django
        django.setup()
        from biomedisa_app.models import Upload
        from biomedisa_app.views import send_error_message
        image = Upload.objects.get(pk=img_id)
        Upload.objects.create(user=image.user, project=image.project, log=1, imageType=None, shortfilename=message)
        send_error_message(image.user.username, image.shortfilename, message)

        # stop processing
        image.path_to_model = ''
        image.status = 0
        image.pid = 0
        image.save()

def create_pid_object(pid, remote=False, queue=None, img_id=None, path_to_model=''):

    # remote server
    if remote:
        with open(biomedisa.BASE_DIR + f'/log/pid_{queue}', 'w') as pidfile:
            print(pid, file=pidfile)

    # local server
    else:
        import django
        django.setup()
        from biomedisa_app.models import Upload
        image = Upload.objects.get(pk=img_id)
        image.path_to_model = path_to_model
        image.pid = pid
        image.save()

def post_processing(path_to_final, time_str, server_name, remote, queue, dice=1.0, path_to_model=None, path_to_uq=None, path_to_smooth=None, path_to_cropped_image=None, uncertainty=False, smooth=False, img_id=None, label_id=None, train=False, predict=False):

    # remote server
    if remote:
        with open(biomedisa.BASE_DIR + f'/log/config_{queue}', 'w') as configfile:
            print(path_to_final, path_to_uq, path_to_smooth, uncertainty, smooth, str(time_str).replace(' ','-'), server_name, path_to_model, path_to_cropped_image, dice, file=configfile)

    # local server
    else:
        import django
        django.setup()
        from biomedisa_app.models import Upload
        from biomedisa_app.views import send_notification
        from biomedisa.features.active_contour import init_active_contour
        from biomedisa.features.remove_outlier import init_remove_outlier
        from biomedisa.features.create_slices import create_slices
        from redis import Redis
        from rq import Queue

        # get object
        image = Upload.objects.get(pk=img_id)

        if train:
            # create model object
            shortfilename = os.path.basename(path_to_model)
            filename = 'images/' + image.user.username + '/' + shortfilename
            Upload.objects.create(pic=filename, user=image.user, project=image.project, imageType=4, shortfilename=shortfilename)

            # create monitoring objects
            for suffix in ['_acc.png', '_loss.png', '.csv']:
                shortfilename = os.path.basename(path_to_model.replace('.h5', suffix))
                filename = 'images/' + image.user.username + '/' + shortfilename
                Upload.objects.create(pic=filename, user=image.user, project=image.project, imageType=6, shortfilename=shortfilename)

        else:
            # create final objects
            shortfilename = os.path.basename(path_to_final)
            filename = 'images/' + image.user.username + '/' + shortfilename
            final = Upload.objects.create(pic=filename, user=image.user, project=image.project, final=1, active=1, imageType=3, shortfilename=shortfilename)
            final.friend = final.id
            final.save()

            if path_to_cropped_image:
                shortfilename = os.path.basename(path_to_cropped_image)
                filename = 'images/' + image.user.username + '/' + shortfilename
                Upload.objects.create(pic=filename, user=image.user, project=image.project, final=9, active=0, imageType=3, shortfilename=shortfilename, friend=final.id)

            if uncertainty:
                shortfilename = os.path.basename(path_to_uq)
                filename = 'images/' + image.user.username + '/' + shortfilename
                uncertainty_obj = Upload.objects.create(pic=filename, user=image.user, project=image.project, final=4, imageType=3, shortfilename=shortfilename, friend=final.id)

            if smooth:
                shortfilename = os.path.basename(path_to_smooth)
                filename = 'images/' + image.user.username + '/' + shortfilename
                smooth_obj = Upload.objects.create(pic=filename, user=image.user, project=image.project, final=5, imageType=3, shortfilename=shortfilename, friend=final.id)

            # create allaxes warning
            if dice < 0.3:
                Upload.objects.create(user=image.user, project=image.project,
                    log=1, imageType=None, shortfilename='Bad result! Activate "All axes" if you labeled axes other than the xy-plane.')

            # acwe
            if not (os.path.splitext(path_to_final)[1]=='.tar' or path_to_final[-7:]=='.tar.gz'):
                q = Queue('acwe', connection=Redis())
                job = q.enqueue_call(init_active_contour, args=(img_id, final.id, label_id, True,), timeout=-1)
                job = q.enqueue_call(init_active_contour, args=(img_id, final.id, label_id,), timeout=-1)

            # cleanup
            if not (os.path.splitext(path_to_final)[1]=='.tar' or path_to_final[-7:]=='.tar.gz'):
                q = Queue('cleanup', connection=Redis())
                job = q.enqueue_call(init_remove_outlier, args=(img_id, final.id, label_id,), timeout=-1)
                if smooth:
                    job = q.enqueue_call(init_remove_outlier, args=(img_id, smooth_obj.id, label_id, False,), timeout=-1)

            # create slices
            q = Queue('slices', connection=Redis())
            job = q.enqueue_call(create_slices, args=(image.pic.path, final.pic.path,), timeout=-1)
            if path_to_cropped_image:
                q = Queue('slices', connection=Redis())
                job = q.enqueue_call(create_slices, args=(path_to_cropped_image, None,), timeout=-1)
            if smooth:
                job = q.enqueue_call(create_slices, args=(image.pic.path, smooth_obj.pic.path,), timeout=-1)
            if uncertainty:
                job = q.enqueue_call(create_slices, args=(uncertainty_obj.pic.path, None,), timeout=-1)

        # send notification
        send_notification(image.user.username, image.shortfilename, time_str, server_name, train, predict)

        # stop processing
        image.path_to_model = ''
        image.status = 0
        image.pid = 0
        image.save()

