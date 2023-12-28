##########################################################################
##                                                                      ##
##  Copyright (c) 2023 Philipp Lösel. All rights reserved.              ##
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
from biomedisa.settings import BASE_DIR
from biomedisa_app.config import config

def create_error_object(img_id, message, host=False):
    if host:
        host = config['FIRST_QUEUE_HOST']
        host_base = BASE_DIR
        if 'FIRST_QUEUE_BASE_DIR' in config:
            host_base = config['FIRST_QUEUE_BASE_DIR']
        cmd = ['ssh', f'{host}:{host_base}/biomedisa_features/django_env.py', '-ceo', img_id, message]
        p = subprocess.Popen(cmd)
        p.wait()
    else:
        import django
        django.setup()
        from biomedisa_app.models import Upload
        from biomedisa_app.views import send_error_message
        image = Upload.objects.get(pk=img_id)
        Upload.objects.create(user=image.user, project=image.project, log=1, imageType=None, shortfilename=message)
        send_error_message(image.user.username, image.shortfilename, message)

def create_pid_object(img_id, pid, host=False):
    if host:
        host = config['FIRST_QUEUE_HOST']
        host_base = BASE_DIR
        if 'FIRST_QUEUE_BASE_DIR' in config:
            host_base = config['FIRST_QUEUE_BASE_DIR']
        cmd = ['ssh', f'{host}:{host_base}/biomedisa_features/django_env.py', '-cpo', img_id, pid]
        p = subprocess.Popen(cmd)
        p.wait()
    else:
        import django
        django.setup()
        from biomedisa_app.models import Upload
        image = Upload.objects.get(pk=img_id)
        image.pid = pid
        image.save()

def post_processing(img_id, label_id, path_to_final, path_to_uq, path_to_smooth, uncertainty, smooth, time_str, server_name, username, host=False):

    if host:

        # get host info
        host = config['FIRST_QUEUE_HOST']
        host_base = BASE_DIR
        if 'FIRST_QUEUE_BASE_DIR' in config:
            host_base = config['FIRST_QUEUE_BASE_DIR']
        src_dir = BASE_DIR + f'/private_storage/images/{username}/'
        dst_dir = f'{host}:{host_base}/private_storage/images/{username}/'

        # sync results
        sync_cmd = ['rsync', '-avP', src_dir+path_to_final, dst_dir]
        p = subprocess.Popen(sync_cmd)
        p.wait()
        if uncertainty:
            sync_cmd = ['rsync', '-avP', src_dir+path_to_uq, dst_dir]
            p = subprocess.Popen(sync_cmd)
            p.wait()
        if smooth:
            sync_cmd = ['rsync', '-avP', src_dir+path_to_smooth, dst_dir]
            p = subprocess.Popen(sync_cmd)
            p.wait()

        # post processing
        cmd = ['ssh', f'{host}:{host_base}/biomedisa_features/django_env.py', '-pp']
        cmd += [img_id, label_id, path_to_final, path_to_uq, path_to_smooth, uncertainty, smooth, time_str, server_name, username]
        p = subprocess.Popen(cmd)
        p.wait()

    else:

        import django
        django.setup()
        from biomedisa_app.models import Upload
        from biomedisa_app.views import send_notification
        from biomedisa_features.active_contour import active_contour
        from biomedisa_features.remove_outlier import remove_outlier
        from biomedisa_features.create_slices import create_slices
        from redis import Redis
        from rq import Queue

        # get object
        image = Upload.objects.get(pk=img_id)

        # create final objects
        filename = 'images/' + username + '/' + path_to_final
        final = Upload.objects.create(pic=filename, user=image.user, project=image.project, final=1, active=1, imageType=3, shortfilename=path_to_final)
        final.friend = final.id
        final.save()
        if uncertainty:
            filename = 'images/' + username + '/' + path_to_uq
            uncertainty_obj = Upload.objects.create(pic=filename, user=image.user, project=image.project, final=4, imageType=3, shortfilename=path_to_uq, friend=final.id)
        if smooth:
            filename = 'images/' + username + '/' + path_to_smooth
            smooth_obj = Upload.objects.create(pic=filename, user=image.user, project=image.project, final=5, imageType=3, shortfilename=path_to_smooth, friend=final.id)

        # send notification
        send_notification(username, image.shortfilename, time_str, server_name)

        # acwe
        q = Queue('acwe', connection=Redis())
        job = q.enqueue_call(active_contour, args=(img_id, final.id, label_id, True,), timeout=-1)
        job = q.enqueue_call(active_contour, args=(img_id, final.id, label_id,), timeout=-1)

        # cleanup
        q = Queue('cleanup', connection=Redis())
        job = q.enqueue_call(remove_outlier, args=(img_id, final.id, final.id, label_id,), timeout=-1)
        if smooth:
            job = q.enqueue_call(remove_outlier, args=(img_id, smooth_obj.id, final.id, label_id, False,), timeout=-1)

        # create slices
        q = Queue('slices', connection=Redis())
        job = q.enqueue_call(create_slices, args=(image.pic.path, final.pic.path,), timeout=-1)
        if smooth:
            job = q.enqueue_call(create_slices, args=(image.pic.path, smooth_obj.pic.path,), timeout=-1)
        if uncertainty:
            job = q.enqueue_call(create_slices, args=(uncertainty_obj.pic.path, None,), timeout=-1)

if __name__ == '__main__':

    if sys.argv[1] == '-pp':
        img_id, label_id, path_to_final, path_to_uq, path_to_smooth, uncertainty, smooth, time_str, server_name = sys.argv[2:]
        post_processing(img_id, label_id, path_to_final, path_to_uq, path_to_smooth, uncertainty, smooth, time_str, server_name, username, True)
    elif sys.argv[1] == '-cpo':
        img_id, pid = sys.argv[2:]
        create_pid_object(img_id, pid, True)
    elif sys.argv[1] == '-ceo':
        img_id, message = sys.argv[2:]
        create_error_object(img_id, message, True)

