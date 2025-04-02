#!/usr/bin/python3
##########################################################################
##                                                                      ##
##  Copyright (c) 2019-2025 Philipp Lösel. All rights reserved.         ##
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
from biomedisa.features.biomedisa_helper import (load_data, save_data, unique_file_path,
    img_to_uint8, smooth_img_3x3, _error_)
from biomedisa.features.create_slices import create_slices
from shutil import copytree
import numpy as np
import argparse
import traceback
import subprocess
import time

def init_process_image(id, process=None):

    import django
    django.setup()
    from biomedisa_app.models import Upload
    from biomedisa_app.config import config
    from biomedisa_app.views import send_data_to_host
    from biomedisa.features.django_env import create_error_object
    from redis import Redis
    from rq import Queue

    # get object
    try:
        img = Upload.objects.get(pk=id)
    except Upload.DoesNotExist:
        img.status = 0
        img.save()
        Upload.objects.create(user=img.user, project=img.project,
            log=1, imageType=None, shortfilename='File has been removed.')

    # get host information
    host = ''
    host_base = biomedisa.BASE_DIR
    subhost, qsub_pid = None, None
    if 'REMOTE_QUEUE_HOST' in config:
        host = config['REMOTE_QUEUE_HOST']
    if host and 'REMOTE_QUEUE_SUBHOST' in config:
        subhost = config['REMOTE_QUEUE_SUBHOST']
    if host and 'REMOTE_QUEUE_BASE_DIR' in config:
        host_base = config['REMOTE_QUEUE_BASE_DIR']

    # check if aborted
    if img.status > 0:

        if process=='smooth' and img.imageType!=1:
            Upload.objects.create(user=img.user, project=img.project, log=1,
                imageType=None, shortfilename='No valid image data.')

        else:
            # set status to processing
            img.status = 2
            img.save()

            # suffix
            if process == 'convert':
                suffix = '.8bit.tif'
            elif process == 'smooth':
                suffix = '.denoised.tif'

            # create path to result
            filename, extension = os.path.splitext(img.pic.path)
            if extension == '.gz':
                extension = '.nii.gz'
                filename = filename[:-4]
            path_to_result = unique_file_path(filename + suffix)
            new_short_name = os.path.basename(path_to_result)
            pic_path = 'images/%s/%s' %(img.user.username, new_short_name)

            # remote server
            if host:

                # base command
                qsub, sbatch = False, False
                cmd = ['python3', '-m', 'biomedisa.features.process_image']
                if 'REMOTE_QUEUE_SBATCH' in config and config['REMOTE_QUEUE_SBATCH']:
                    cmd = ['sbatch', 'queue_5.sh']
                    sbatch = True
                elif 'REMOTE_QUEUE_QSUB' in config and config['REMOTE_QUEUE_QSUB']:
                    cmd = []
                    qsub = True

                cmd += [img.pic.path.replace(biomedisa.BASE_DIR,host_base)]
                cmd += [f'-iid={img.id}', '-r', '-q=5']

                # command (append only on demand)
                if process == 'convert':
                    cmd += ['-c']
                elif process == 'smooth':
                    cmd += ['-s']

                # create user directory
                subprocess.Popen(['ssh', host, 'mkdir', '-p', host_base+'/private_storage/images/'+img.user.username]).wait()

                # send data to host
                success = send_data_to_host(img.pic.path, host+':'+img.pic.path.replace(biomedisa.BASE_DIR,host_base))

                # check if aborted
                img = Upload.objects.get(pk=img.id)
                if img.status==2 and img.queue==5 and success==0:

                    # adjust command
                    if qsub:
                        args = " ".join(cmd)
                        cmd = [f"qsub -v ARGS='{args}' queue_5.sh"]
                    if subhost:
                        cmd_host = ['ssh', host, 'ssh', subhost]
                        cmd = cmd_host + cmd
                    else:
                        cmd_host = ['ssh', host]
                        cmd = cmd_host + cmd

                    # config files
                    error_path = '/log/error_5'
                    pid_path = '/log/pid_5'

                    # result path on host
                    result_on_host = img.pic.path.replace(biomedisa.BASE_DIR,host_base)
                    result_on_host = result_on_host.replace(os.path.splitext(result_on_host)[1], suffix)

                    # remove data from host
                    subprocess.Popen(['ssh', host, 'rm', result_on_host]).wait()

                    # submit job
                    if qsub or sbatch:
                        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
                        stdout, stderr = process.communicate()
                        if sbatch:
                            job_id = stdout.strip().split()[-1]
                        else:
                            job_id = stdout.strip().split('.')[0]
                        print(f"submit output: {stdout.strip()}")
                        img.pid = job_id
                        img.save()

                        # wait for the server to finish
                        error, success, started, processing = 1, 1, 1, True
                        while error!=0 and success!=0 and processing:
                            time.sleep(30)
                            error = subprocess.Popen(['scp', host+':'+host_base+error_path, biomedisa.BASE_DIR + error_path]).wait()
                            started = subprocess.Popen(['scp', host +':'+host_base+pid_path, biomedisa.BASE_DIR + pid_path]).wait()
                            success = subprocess.Popen(['scp', host+':'+result_on_host, path_to_result]).wait()
                            img = Upload.objects.filter(pk=img.id).first()
                            if img and img.status==2 and img.queue==5:
                                if started==0 and img.message != 'Processing':
                                    img.message = 'Processing'
                                    img.save()
                            else:
                                processing = False

                    # interactive shell
                    else:
                        process = subprocess.Popen(cmd)
                        img.message = 'Processing'
                        img.pid = process.pid
                        img.save()
                        process.wait()
                        error = subprocess.Popen(['scp', host+':'+host_base+error_path, biomedisa.BASE_DIR + error_path]).wait()
                        started = subprocess.Popen(['scp', host+':'+host_base+pid_path, biomedisa.BASE_DIR + pid_path]).wait()
                        success = subprocess.Popen(['scp', host+':'+result_on_host, path_to_result]).wait()

                    # create error object
                    if error == 0:
                        with open(biomedisa.BASE_DIR + error_path, 'r') as errorfile:
                            message = errorfile.read()
                        create_error_object(message, img_id=img.id)

                    # create result object
                    elif success == 0:
                        active = 1 if img.imageType == 3 else 0
                        Upload.objects.create(pic=pic_path, user=img.user, project=img.project,
                            imageType=img.imageType, shortfilename=new_short_name, active=active)

                    # remove config files
                    subprocess.Popen(['ssh', host, 'rm', host_base + error_path]).wait()
                    subprocess.Popen(['ssh', host, 'rm', host_base + pid_path]).wait()

            # local server
            else:

                # set pid and processing status
                img.pid = int(os.getpid())
                img.message = 'Processing'
                img.save()

                # load data
                data, header = load_data(img.pic.path)
                if data is None:
                    # return error
                    success = 1
                    Upload.objects.create(user=img.user, project=img.project,
                        log=1, imageType=None, shortfilename='Invalid data.')
                else:
                    # process data
                    success = 0
                    if process == 'convert':
                        data = img_to_uint8(data)
                        save_data(path_to_result, data, final_image_type='.tif')
                    elif process == 'smooth':
                        data = smooth_img_3x3(data)
                        save_data(path_to_result, data, final_image_type='.tif')

                    # create object
                    active = 1 if img.imageType == 3 else 0
                    Upload.objects.create(pic=pic_path, user=img.user, project=img.project,
                        imageType=img.imageType, shortfilename=new_short_name, active=active)

            # copy or create slices for preview
            if success==0 and process == 'convert':
                path_to_source = img.pic.path.replace('images', 'sliceviewer', 1)
                path_to_dest = path_to_result.replace('images', 'sliceviewer', 1)
                if os.path.exists(path_to_source) and not os.path.exists(path_to_dest):
                    copytree(path_to_source, path_to_dest, copy_function=os.link)
            elif success==0 and process == 'smooth':
                q = Queue('slices', connection=Redis())
                job = q.enqueue_call(create_slices, args=(path_to_result, None,), timeout=-1)

        # close process
        img.status = 0
        img.pid = 0
        img.save()

if __name__ == "__main__":

    # initialize arguments
    parser = argparse.ArgumentParser(description='Biomedisa process image.',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required arguments
    parser.add_argument('path_to_data', type=str, metavar='PATH_TO_DATA',
                        help='Location of image data')

    # optional arguments
    parser.add_argument('-c', '--convert', action='store_true', default=False,
                        help='Convert to uint8 TIFF')
    parser.add_argument('-s', '--smooth', action='store_true', default=False,
                        help='Denoise/smooth image data')
    parser.add_argument('-iid','--img_id', type=str, default=None,
                        help='Image ID within django environment/browser version')
    parser.add_argument('-r','--remote', action='store_true', default=False,
                        help='Process is carried out on a remote server. Must be set up in config.py')
    parser.add_argument('-q','--queue', type=int, default=0,
                        help='Processing queue when using a remote server')
    bm = parser.parse_args()

    # set pid
    if bm.remote:
        from biomedisa.features.django_env import create_pid_object
        create_pid_object(os.getpid(), True, 5, bm.img_id)

    # django environment
    if bm.img_id is not None:
        bm.django_env = True
        bm.username = os.path.basename(os.path.dirname(bm.path_to_data))
        bm.shortfilename = os.path.basename(bm.path_to_data)
        bm.path_to_logfile = biomedisa.BASE_DIR + '/log/logfile.txt'
    else:
        bm.django_env = False

    # load data
    if bm.convert or bm.smooth:
        bm.image, _ = load_data(bm.path_to_data)
        if bm.image is None:
            bm = _error_(bm, 'Invalid data.')
        else:
            try:
                # suffix
                if bm.convert:
                    bm.image = img_to_uint8(bm.image)
                    suffix = '.8bit.tif'
                elif bm.smooth:
                    bm.image = smooth_img_3x3(bm.image)
                    suffix = '.denoised.tif'

                # save result
                path_to_result = bm.path_to_data.replace(os.path.splitext(bm.path_to_data)[1], suffix)
                save_data(path_to_result, bm.image, final_image_type='.tif')

            except Exception as e:
                print(traceback.format_exc())
                bm = _error_(bm, str(e))

