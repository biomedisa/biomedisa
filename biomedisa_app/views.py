##########################################################################
##                                                                      ##
##  Copyright (c) 2020 Philipp Lösel. All rights reserved.              ##
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

# 04. hash_a_string(string)
# 05. get_size(start_path)
# 06. index(request)
# 07. faq(request)
# 08. contact(request)
# 09. impressum(request)
# 10. publications(request)
# 11. logout_user(request)
# 12. sendEmail(datas)
# 13. generate_activation_key()
# 14. register(request)
# 15. activation(request, key)
# 16. change_active_final(request, id, val)
# 17. sliceviewer(request, id)
# 18. visualization(request, id)
# 19. login_user(request)
# 20. settings(request, id)
# 21. update_profile(request)
# 22. change_password(request)
# 23. storage(request)
# 24. move(request, id, project)
# 25. features(request, action)
# 26. reset(request)
# 27. app(request)
# 28. constant_time_compare(val1, val2)
# 29. share_data(request, id, list_of_users, next)
# 30. clean_state(request, next)
# 31. create_download_link(request, id)
# 32. accept_shared_data(request, id)
# 33. download_shared_data(request, id, pw)
# 34. download(request, id)
# 35. gallery(request)
# 36. run_demo(request)
# 37. delete_demo(request)
# 38. download_demo(request, id)
# 39. visualization_demo(request, id)
# 40. sliceviewer_demo(request, id)
# 41. delete(request, id)
# 42. init_random_walk(image, label)
# 43. run(request)
# 44. stop_running_job(id)
# 45. remove_from_queue(request, id) or kill process
# 46. delete account(request)
# 47. send_notification(username, image_name, time_str, server_name, train=False)
# 48. send_start_notification(image, remote=False)
# 49. send_error_message(username, image_name, error_msg)
# 50. send_share_notify(username, image_name, shared_by)
# 51. status(request)
# 52. dummy(request)
#
##################################################

from __future__ import unicode_literals

import django
django.setup()
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, StreamingHttpResponse, JsonResponse
from django.template.context_processors import csrf
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.urls import reverse
from django.contrib.auth.models import User
from django.contrib.auth.forms import PasswordChangeForm
from django.template import Template, Context
from django.core.mail import send_mail
from django.utils import timezone

from biomedisa_app.models import (UploadForm, Upload, StorageForm, Profile,
    UserForm, SettingsForm, SettingsPredictionForm, CustomUserCreationForm)
from biomedisa_features.create_slices import create_slices
from biomedisa_features.biomedisa_helper import (load_data, save_data, img_to_uint8, id_generator,
    convert_image, smooth_image)
from django.utils.crypto import get_random_string
from biomedisa_app.config import config
from multiprocessing import Process

from wsgiref.util import FileWrapper
import numpy as np
from PIL import Image
import os, sys
import time
import json
from decimal import Decimal
import hashlib
from shutil import copytree
import tarfile, zipfile
import shutil
import subprocess

# list of demo files
pk_list = [3856, 3857, 14244, \
           36600, 36601, 36602, \
           3860, 3861, 14253, \
           8321, 14257, 14258, \
           8315, 8322, 14261, \
           21089, 21090, 21091, \
           21088, 21140, \
           36666,36667,36668, \
           36645,36646,36659, \
           36647,36648,36687, \
           36650,36651,36652, \
           36716,36717,36728]

if config['OS'] == 'linux':
    from redis import Redis
    from rq import Queue

# 02. maintenance
def maintenance(request):
    return render(request, 'maintenance.html')

# 03. partners
def partners(request):
    return render(request, 'partners.html')

# 04. hash_a_string
def hash_a_string(string):
    return hashlib.sha256(string.encode('utf-8')).hexdigest()

# 05. get_size
def get_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

# 06. index
def index(request):
    request.session['k'] = 0
    return render(request, 'index.html')

# 07. faq
def faq(request):
    return render(request, 'faq.html')

# 08. contact
def contact(request):
    return render(request, 'contact.html')

# 09. impressum
def impressum(request):
    return render(request, 'impressum.html')

# 10. publications
def publications(request):
    return render(request, 'publications.html')

# 11. logout_user
@login_required
def logout_user(request):
    # remove symbolic links created for sliceviewer and visualization
    try:
        symlinks = request.session["symlinks"]
        del request.session["symlinks"]
        for symlink in symlinks:
            os.unlink(symlink)
    except:
        pass
    logout(request)
    return redirect('index')

# 12. send email notification
def sendEmail(datas):
    c = datas['context']
    f = open(config['PATH_TO_BIOMEDISA'] + '/biomedisa_app/messages' + datas['email_path'], 'r')
    t = Template(f.read())
    f.close()
    message = t.render(c)
    send_mail(datas['email_subject'], message, 'Biomedisa <%s>' %(config['EMAIL']), [datas['email']], fail_silently=True)

# 13. create random string
def generate_activation_key():
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    secret_key = get_random_string(20, chars)
    return secret_key

# 14. register
def register(request):
    if request.method == 'POST':
        f = CustomUserCreationForm(request.POST)
        if f.is_valid():
            cd = f.cleaned_data

            # prepare data package
            datas={}
            key = generate_activation_key()
            datas['email'] = cd['email']
            datas['is_active'] = not config['EMAIL_CONFIRMATION']
            datas['activation_key'] = hash_a_string(key)
            datas['key_expires'] = timezone.now() + timezone.timedelta(days=2)
            datas['email_path'] = '/ActivationEmail.txt'
            datas['email_subject'] = 'Activate your Biomedisa account'
            link = config['SERVER'] + '/activate/' + key
            datas['context'] = Context({'activation_link':link, 'username':cd['username'],
                               'key_expires':datas['key_expires'].strftime('%Y-%m-%d %H:%M:%S')})

            # return error if spam
            if cd['email'][-3:] == '.ru' or cd['institution'] == 'REG':
                some_integer = int('some_string')
            if any(['sex' in some_string for some_string in [cd['institution'], cd['subject'], cd['message']]]):
                some_integer = int('some_string')

            # create user
            meta = f.save(datas)

            # send confirmation link or activate account immediately
            if config['EMAIL_CONFIRMATION']:
                sendEmail(datas)
                messages.success(request, 'We have sent an email with a confirmation link to your email address.')
            else:
                # create user directories
                for directory in ['images', 'dataset', 'sliceviewer']:
                    path_to_data = config['PATH_TO_BIOMEDISA'] + '/private_storage/' + directory + '/' + meta['username']
                    os.makedirs(path_to_data)
                    try:
                        os.chmod(path_to_data, 0o770)
                    except:
                        pass
                messages.success(request, 'Account created successfully.')

            # write in biomedisa/log/applications.txt
            afile = open(config['PATH_TO_BIOMEDISA'] + '/log/applications.txt', 'a')
            msg = 'Time: %s \nEmail: %s \nUsername: %s \nInstitution: %s \nSubject: %s \nMessage: %s' \
                %(time.ctime(), meta['email'], meta['username'], meta['institution'].encode('ascii', 'ignore').decode(), \
                meta['subject'].encode('ascii', 'ignore').decode(), meta['message'].encode('ascii', 'ignore').decode())
            afile.write(msg)
            afile.write('\n \n---------------------------------\n \n')
            afile.close()

            # prepare data package
            datas={}
            datas['email'] = config['EMAIL']
            datas['email_path'] = '/MessageEmail.txt'
            datas['email_subject'] = 'Application'
            datas['context'] = Context({'msg':msg})

            # send notification to admin
            sendEmail(datas)

            return redirect('register')

    else:
        f = CustomUserCreationForm()

    return render(request, 'register.html', {'form': f})

# 15. activate user account
def activation(request, key):
    key = str(key)
    key = hash_a_string(key)
    profile = get_object_or_404(Profile, activation_key=key)
    content = {}
    if profile.user.is_active == False:
        if timezone.now() > profile.key_expires:
            content['msg'] = 'The activation link has expired.'
            content['sub_msg'] = 'Please contact us for an activation.'
        else:
            content['msg'] = 'Account successfully activated.'
            profile.user.is_active = True
            profile.user.save()
            # create user directories
            for directory in ['images', 'dataset', 'sliceviewer']:
                path_to_data = config['PATH_TO_BIOMEDISA'] + '/private_storage/' + directory + '/' + str(profile.user.username)
                os.makedirs(path_to_data)
                try:
                    os.chmod(path_to_data, 0o770)
                except:
                    pass
    else:
        content['msg'] = 'Your account has already been activated.'
    return render(request, 'contact.html', content)

# 16. change_active_final
@login_required
def change_active_final(request, id, val):
    id = int(id)
    val = int(val)
    stock_to_change = get_object_or_404(Upload, pk=id)
    if stock_to_change.user == request.user and stock_to_change.friend:
        changed = False
        images = Upload.objects.filter(user=request.user, friend=stock_to_change.friend)
        for img in images:
            if img.user == request.user and img.final == val:
                stock_to_change.active = 0
                stock_to_change.save()
                img.active = 1
                img.save()
                changed = True
        if not changed:
            request.session['state'] = "file does not exist"
    next = request.GET.get('next', '')
    next = str(next)
    if next == "storage":
        return redirect(storage)
    else:
        return redirect(app)

# 17. sliceviewer
@login_required
def sliceviewer(request, id):
    id = int(id)
    stock_to_show = get_object_or_404(Upload, pk=id)
    if stock_to_show.user == request.user:

        prefix = generate_activation_key()
        path_to_slices = '/media/' + prefix
        src = config['PATH_TO_BIOMEDISA'] + '/private_storage/sliceviewer/%s/%s' %(request.user, stock_to_show.shortfilename)
        dest = config['PATH_TO_BIOMEDISA'] + path_to_slices
        os.symlink(src, dest)
        path_to_slices += '/'

        # create symlinks wich are removed when "app" is called or user loggs out
        try:
            symlinks = request.session["symlinks"]
            symlinks.append(dest)
            request.session["symlinks"] = symlinks
        except:
            request.session["symlinks"] = [dest]

        if os.path.isdir(dest):
            nos = len(os.listdir(dest)) - 1
            im = Image.open(dest + '/0.png')
            imshape = np.asarray(im).shape
            return render(request, 'sliceviewer.html', {'path_to_slices':path_to_slices, 'nos':nos, 'imshape_x':imshape[1], 'imshape_y':imshape[0]})
        else:
            request.session['state'] = 'Slices are not yet available.'
            next = request.GET.get('next', '')
            next = str(next)
            if next == 'storage':
                return redirect(storage)
            else:
                return redirect(app)

# 18. visualization
@login_required
def visualization(request, id):
    id = int(id)
    stock_to_show = get_object_or_404(Upload, pk=id)
    if stock_to_show.user == request.user:

        prefix = generate_activation_key()
        path_to_slices = '/media/' + prefix
        src = config['PATH_TO_BIOMEDISA'] + '/private_storage/dataset/%s/%s' %(request.user, stock_to_show.shortfilename)
        dest = config['PATH_TO_BIOMEDISA'] + path_to_slices
        os.symlink(src, dest)

        # create symlinks wich are removed when "app" is called or user loggs out
        try:
            symlinks = request.session["symlinks"]
            symlinks.append(dest)
            request.session["symlinks"] = symlinks
        except:
            request.session["symlinks"] = [dest]

        if os.path.isdir(dest):
            noslices = len(os.listdir(dest))
            path_to_slicemaps = [path_to_slices + '/slicemap_%s.png' %(k) for k in range(noslices)]
            path_to_slicemaps = json.dumps(path_to_slicemaps)
            return render(request, 'visualization.html', {'path_to_slicemaps':path_to_slicemaps})
        else:
            request.session['state'] = 'Visualization is not yet available.'
            next = request.GET.get('next', '')
            next = str(next)
            if next == 'storage':
                return redirect(storage)
            else:
                return redirect(app)

# 19. login_user
def login_user(request):
    c = {}
    c.update(csrf(request))
    state = ''
    username = ''
    password = ''
    if request.POST:
        username = request.POST.get('username')
        password = request.POST.get('password')
        username = str(username)
        password = str(password)
        tmp = username.split('::')
        if len(tmp) > 1:
            username = tmp[0]

        if '@' in username:
            try:
                user_id = User.objects.get(email=username)
                username = user_id.username
            except:
                pass

        user = authenticate(username=username, password=password)

        if user is not None:
            if user.is_superuser and len(tmp) > 1:
                try:
                    user = User.objects.get(username=tmp[1])
                except:
                    pass
            if user.is_active:
                login(request, user)
                return redirect('/app/')
        else:
            state = 'Your username, e-mail or password were incorrect.'
    return render(request, ('auth.html', c), {'state': state, 'username': username})

# 20. settings
@login_required
def settings(request, id):
    id = int(id)
    image = get_object_or_404(Upload, pk=id)
    if image.user == request.user:
        if request.method == 'POST':
            img = SettingsForm(request.POST)
            if img.is_valid():
                cd = img.cleaned_data
                tmp0 = image.delete_outliers
                tmp1 = image.fill_holes
                tmp2 = image.allaxis
                tmp3 = image.normalize
                tmp4 = image.smooth
                tmp5 = image.compression
                tmp6 = image.ac_alpha
                tmp7 = image.ac_smooth
                tmp8 = image.ac_steps
                tmp9 = image.ignore
                tmp10 = image.epochs
                tmp11 = image.uncertainty
                tmp13 = image.only
                tmp14 = image.position
                tmp15 = image.stride_size
                tmp16 = image.batch_size
                tmp18 = image.x_scale
                tmp19 = image.y_scale
                tmp20 = image.z_scale
                tmp21 = image.balance
                tmp22 = image.flip_x
                tmp23 = image.flip_y
                tmp24 = image.flip_z
                tmp25 = image.rotate
                image.allaxis = cd['allaxis']
                image.smooth = cd['smooth']
                image.ac_alpha = cd['ac_alpha']
                image.ac_smooth = cd['ac_smooth']
                image.ac_steps = cd['ac_steps']
                image.delete_outliers = cd['delete_outliers']
                image.fill_holes = cd['fill_holes']
                image.ignore = cd['ignore']
                image.epochs = min(200, int(cd['epochs']))
                image.uncertainty = cd['uncertainty']
                image.normalize = cd['normalize']
                image.compression = cd['compression']
                image.only = cd['only']
                image.position = cd['position']
                image.stride_size = cd['stride_size']
                image.batch_size = cd['batch_size']
                image.x_scale = cd['x_scale']
                image.y_scale = cd['y_scale']
                image.z_scale = cd['z_scale']
                image.balance = cd['balance']
                image.flip_x = cd['flip_x']
                image.flip_y = cd['flip_y']
                image.flip_z = cd['flip_z']
                image.rotate = min(180, max(0, int(cd['rotate'])))
                image.save()

                if any([tmp0!=image.delete_outliers, tmp1!=image.fill_holes, tmp2!=image.allaxis, tmp3!=image.normalize,
                        tmp4!=image.smooth, tmp5!=image.compression, tmp6!=image.ac_alpha, tmp7!=image.ac_smooth,
                        tmp8!=image.ac_steps, tmp9!=image.ignore, tmp10!=image.epochs, tmp11!=image.uncertainty,
                        tmp13!=image.only, tmp14!=image.position, tmp15!=image.stride_size, tmp16!=image.batch_size,
                        tmp18!=image.x_scale, tmp19!=image.y_scale, tmp20!=image.z_scale, tmp21!=image.balance,
                        tmp22!=image.flip_x, tmp23!=image.flip_y, tmp24!=image.flip_z, tmp25!=image.rotate]):
                    messages.error(request, 'Your settings were changed.')

                return redirect(settings, id)
        else:
            settings_form = SettingsForm(initial={'allaxis':image.allaxis, 'smooth':image.smooth, 'ac_alpha':image.ac_alpha,
                                'ac_smooth':image.ac_smooth, 'ac_steps':image.ac_steps, 'delete_outliers':image.delete_outliers,
                                'fill_holes':image.fill_holes, 'ignore':image.ignore, 'epochs':image.epochs, 'uncertainty':image.uncertainty,
                                'normalize':image.normalize, 'position':image.position, 'compression':image.compression,
                                'only':image.only, 'stride_size':image.stride_size, 'batch_size':image.batch_size,
                                'x_scale':image.x_scale, 'y_scale':image.y_scale, 'z_scale':image.z_scale, 'balance':image.balance,
                                'flip_x':image.flip_x, 'flip_y':image.flip_y, 'flip_z':image.flip_z, 'rotate':image.rotate})

            return render(request, 'settings.html', {'settings_form':settings_form, 'name':image.shortfilename})

# 20. settings
@login_required
def settings_prediction(request, id):
    id = int(id)
    image = get_object_or_404(Upload, pk=id)
    if image.user == request.user and image.imageType == 4 and image.project > 0:
        if request.method == 'POST':
            img = SettingsPredictionForm(request.POST)
            if img.is_valid():
                cd = img.cleaned_data
                tmp0 = image.delete_outliers
                tmp1 = image.fill_holes
                tmp2 = image.compression
                tmp3 = image.stride_size
                tmp4 = image.ac_alpha
                tmp5 = image.ac_smooth
                tmp6 = image.ac_steps
                tmp7 = image.batch_size
                image.delete_outliers = cd['delete_outliers']
                image.fill_holes = cd['fill_holes']
                image.compression = cd['compression']
                image.stride_size = cd['stride_size']
                image.ac_alpha = cd['ac_alpha']
                image.ac_smooth = cd['ac_smooth']
                image.ac_steps = cd['ac_steps']
                image.batch_size = cd['batch_size']
                image.save()
                if any([tmp0!=image.delete_outliers, tmp1!=image.fill_holes, tmp2!=image.compression,\
                        tmp3!=image.stride_size, tmp4!=image.ac_alpha, tmp5!=image.ac_smooth, \
                        tmp6!=image.ac_steps, tmp7!=image.batch_size]):
                    messages.error(request, 'Your settings were changed.')
                return redirect(settings_prediction, id)
        else:
            settings_form = SettingsPredictionForm(initial={'delete_outliers':image.delete_outliers, \
                                'fill_holes':image.fill_holes, 'compression':image.compression, \
                                'stride_size':image.stride_size, 'ac_alpha':image.ac_alpha, \
                                'ac_smooth':image.ac_smooth, 'ac_steps':image.ac_steps, 'batch_size':image.batch_size})
            return render(request, 'settings.html', {'settings_form':settings_form, 'name':image.shortfilename})

# 21. update_profile
@login_required
def update_profile(request):
    profile = Profile.objects.get(user=request.user)
    if request.method == 'POST':
        user_form = UserForm(request.POST, instance=request.user)
        if user_form.is_valid():
            cd = user_form.cleaned_data
            profile.notification = cd['notification']
            user_form.save()
            profile.save()
            messages.error(request, 'Your profile was successfully updated!')
            return redirect(update_profile)
        else:
            messages.error(request, 'Please correct the error below.')
    else:
        user_form = UserForm(instance=request.user, initial={'notification':profile.notification})
    return render(request, 'profile.html', {'user_form': user_form})

# 22. change_password
@login_required
def change_password(request):
    if request.method == 'POST':
        pw_form = PasswordChangeForm(request.user, request.POST)
        if pw_form.is_valid():
            user = pw_form.save()
            update_session_auth_hash(request, user)
            messages.success(request, 'Your password was successfully updated!')
            return redirect('change_password')
        else:
            messages.error(request, 'Please correct the error below.')
    else:
        pw_form = PasswordChangeForm(request.user)
    return render(request, 'change_password.html', {'pw_form': pw_form})

# 23. storage
@login_required
def storage(request):

    action = request.GET.get('action')
    if action:
        features(request, action)
        return redirect(storage)

    try:
        state = request.session["state"]
    except KeyError:
        state = None

    if request.method == "POST":
        img = StorageForm(request.POST, request.FILES)
        if img.is_valid():
            newimg = Upload(pic=request.FILES['pic'])
            cd = img.cleaned_data
            newimg.imageType = cd['imageType']
            newimg.project = 0
            newimg.user = request.user
            newimg.save()
            newimg.shortfilename = os.path.basename(newimg.pic.path)
            newimg.save()

            # untar or unzip if necessary
            path_to_dir, extension = os.path.splitext(newimg.pic.path)
            if extension == '.gz':
                path_to_dir, extension = os.path.splitext(path_to_dir)
            if extension == '.zip':
                zip_ref = zipfile.ZipFile(newimg.pic.path, 'r')
                zip_ref.extractall(path=path_to_dir)
                zip_ref.close()
            elif extension == '.tar':
                tar = tarfile.open(newimg.pic.path)
                tar.extractall(path=path_to_dir)
                tar.close()

            return redirect(storage)
    else:
        img = StorageForm()

    # update lists
    images = Upload.objects.filter(user=request.user, project=0)

    # storage size
    storage_size = request.user.profile.storage_size
    datasize = get_size(config['PATH_TO_BIOMEDISA'] + '/private_storage/images/' + request.user.username)
    storage_full = 1 if datasize > storage_size*10e8 else 0
    datasize *= 10e-10
    if datasize < 1:
        datasize = Decimal(datasize)
        datasize = round(datasize, 3)
    else:
        datasize = int(datasize)

    return render(request, 'storage.html', {'state':state, 'images':images, 'form':img, 'datasize':datasize, 'storage_full':storage_full, 'storage_size':storage_size})

# 24. move
@login_required
def move(request):
    results = {'success':False}
    if request.method == 'GET':
        id = request.GET.get('id')
        project = request.GET.get('project')
        id = int(id)
        project = int(project)

        stock_to_move = get_object_or_404(Upload, pk=id)
        if stock_to_move.user == request.user:

            if int(project) == 0:

                if stock_to_move.final:
                    images = Upload.objects.filter(user=request.user, friend=stock_to_move.friend)
                else:
                    images = [stock_to_move]
                for img in images:
                    img.project = project
                    img.save()

                results = {'success':True}

            else:

                #images_tmp = Upload.objects.filter(user=request.user, project=project, imageType=stock_to_move.imageType)
                #if any(images_tmp) and stock_to_move.imageType in [1,2]:
                #    if stock_to_move.imageType == 1:
                #        state = "Image already exists in project %s" %(project)
                #    elif stock_to_move.imageType == 2:
                #        state = "Label already exists in project %s" %(project)
                #   elif stock_to_move.imageType == 3:
                #       state = "Final already exists in project %s" %(project)
                #   elif stock_to_move.imageType == 4:
                #       state = "Network already exists in project %s" %(project)

                #   results = {'success':True, 'msg': state}
                #else:
                if stock_to_move.final:
                    images = Upload.objects.filter(user=request.user, friend=stock_to_move.friend)
                else:
                    images = [stock_to_move]
                for img in images:
                    img.project = project
                    img.save()

                # create slices
                if stock_to_move.imageType == 1:

                    path_to_slices = stock_to_move.pic.path.replace("images", "sliceviewer", 1)
                    if not os.path.exists(path_to_slices):
                        if config['OS'] == 'linux':
                            q = Queue('slices', connection=Redis())
                            job = q.enqueue_call(create_slices, args=(stock_to_move.pic.path, None,), timeout=-1)
                        elif config['OS'] == 'windows':
                            Process(target=create_slices, args=(stock_to_move.pic.path, None)).start()
                    try:
                        image_tmp = Upload.objects.get(user=request.user, project=stock_to_move.project, imageType=2)
                        path_to_slices = image_tmp.pic.path.replace("images", "sliceviewer", 1)
                        if not os.path.exists(path_to_slices):
                            if config['OS'] == 'linux':
                                q = Queue('slices', connection=Redis())
                                job = q.enqueue_call(create_slices, args=(stock_to_move.pic.path, image_tmp.pic.path,), timeout=-1)
                            elif config['OS'] == 'windows':
                                Process(target=create_slices, args=(stock_to_move.pic.path, image_tmp.pic.path)).start()
                    except Upload.DoesNotExist:
                        pass

                elif stock_to_move.imageType == 2:

                    path_to_slices = stock_to_move.pic.path.replace("images", "sliceviewer", 1)
                    if not os.path.exists(path_to_slices):
                        if config['OS'] == 'linux':
                            q = Queue('slices', connection=Redis())
                            job = q.enqueue_call(create_slices, args=(None, stock_to_move.pic.path,), timeout=-1)
                        elif config['OS'] == 'windows':
                            Process(target=create_slices, args=(None, stock_to_move.pic.path)).start()
                    try:
                        image_tmp = Upload.objects.get(user=request.user, project=stock_to_move.project, imageType=1)
                        path_to_slices = stock_to_move.pic.path.replace("images", "sliceviewer", 1)
                        if not os.path.exists(path_to_slices):
                            if config['OS'] == 'linux':
                                q = Queue('slices', connection=Redis())
                                job = q.enqueue_call(create_slices, args=(image_tmp.pic.path, stock_to_move.pic.path,), timeout=-1)
                            elif config['OS'] == 'windows':
                                Process(target=create_slices, args=(image_tmp.pic.path, stock_to_move.pic.path)).start()
                    except Upload.DoesNotExist:
                        pass

                results = {'success':True}

    return JsonResponse(results)

@login_required
def rename_file(request):
    results = {'success':False}
    if request.method == 'GET':
        id = request.GET.get('id')
        new_name = request.GET.get('filename')
        id = int(id)
        new_name = str(new_name)
        stock_to_rename = get_object_or_404(Upload, pk=id)

        # get source file extension
        filename, src_ext = os.path.splitext(stock_to_rename.shortfilename)
        if src_ext == '.gz':
            filename, src_ext = os.path.splitext(filename)
            if src_ext == '.tar':
                src_ext = '.tar.gz'

        # get new filename extension
        new_name = new_name.encode('ascii', 'ignore').decode()
        new_name, extension = os.path.splitext(new_name)
        if extension == '.gz':
            new_name, extension = os.path.splitext(new_name)
            if extension == '.nii':
                extension = '.nii.gz'
            elif extension == '.tar':
                extension = '.tar.gz'

        # make sure filename is not too long
        test_name = 'images/' + stock_to_rename.user.username + '/' + new_name
        maxlen = 100 - len(test_name) - len(extension)
        if maxlen < 0:
            new_name = new_name[:maxlen] + extension
        else:
            new_name = new_name + extension

        if stock_to_rename.user == request.user:
            dirname = os.path.dirname(stock_to_rename.pic.path)
            if os.path.exists(dirname + '/' + new_name):
                state = "File already exists."
                results = {'success':True, 'msg':state}
            else:
                # rename file
                os.rename(stock_to_rename.pic.path, dirname + '/' + new_name)
                # rename extracted .tar or .zip files
                if src_ext in ['.tar','.zip','.tar.gz'] and os.path.exists(stock_to_rename.pic.path[:-len(src_ext)]):
                    os.rename(stock_to_rename.pic.path[:-len(src_ext)], dirname + '/' + new_name[:-len(extension)])
                # rename slices
                path_to_slices = stock_to_rename.pic.path.replace("images", "sliceviewer", 1)
                if os.path.exists(path_to_slices):
                    dirname = os.path.dirname(path_to_slices)
                    os.rename(path_to_slices, dirname + '/' + new_name)
                # rename dataset
                path_to_dataset = stock_to_rename.pic.path.replace("images", "dataset", 1)
                if os.path.exists(path_to_dataset):
                    dirname = os.path.dirname(path_to_dataset)
                    os.rename(path_to_dataset, dirname + '/' + new_name)
                # rename django object
                stock_to_rename.shortfilename = new_name
                stock_to_rename.pic.name = 'images/' + stock_to_rename.user.username + '/' + new_name
                stock_to_rename.save()
                results = {'success':True}
    return JsonResponse(results)

# init_keras_3D
def init_keras_3D(image, label, model, refine, predict, img_list, label_list):

    # get object
    img_obj = get_object_or_404(Upload, pk=image)

    # set status to processing
    if img_obj.status == 1:
        img_obj.status = 2
        img_obj.message = 'Processing'
        img_obj.save()

    # change working directory
    cwd = config['PATH_TO_BIOMEDISA'] + '/biomedisa_features/'

    # get configuration
    if config['THIRD_QUEUE']:
        host = config['THIRD_QUEUE_HOST']
    else:
        host = config['FIRST_QUEUE_HOST']

    # run keras
    if host:
        p = subprocess.Popen(['ssh', host, 'python3', cwd+'keras_3D.py',\
            str(image), str(label), str(model), str(refine), str(predict), str(img_list), str(label_list)])
    else:
        p = subprocess.Popen(['python3', 'keras_3D.py', str(image), str(label),\
            str(model), str(refine), str(predict), str(img_list), str(label_list)], cwd=cwd)

    # wait for process terminating
    p.wait()

# 25. features
def features(request, action):

    # delete files
    if int(action) == 1:
        todo = request.GET.getlist('selected')
        for id in todo:
            id = int(id)
            stock_to_delete = get_object_or_404(Upload, pk=id)
            if stock_to_delete.user == request.user and stock_to_delete.status==0:
                if stock_to_delete.final:
                    images = Upload.objects.filter(user=request.user, friend=stock_to_delete.friend)
                    for img in images:
                        img.delete()
                else:
                    stock_to_delete.delete()

    # move files
    elif int(action) == 2:

        todo = request.GET.getlist('selected')
        for id in todo:
            id = int(id)
            stock_to_move = get_object_or_404(Upload, pk=id)
            if stock_to_move.user == request.user and stock_to_move.status==0:
                if stock_to_move.final:
                    images = Upload.objects.filter(user=request.user, friend=stock_to_move.friend)
                    for img in images:
                        img.project = 0
                        img.save()
                else:
                    stock_to_move.project = 0
                    stock_to_move.save()

    # predict segmentation
    elif int(action) == 3:

        todo = request.GET.getlist('selected')
        images = Upload.objects.filter(pk__in=todo)

        # get models
        model, refine = 0, 0
        for img in images:
            if img.imageType == 4 and str(img.shortfilename[-10:]) != '_refine.h5':
                model = img.id
            elif img.imageType == 4 and str(img.shortfilename[-10:]) == '_refine.h5':
                refine = img.id

        # predict segmentation
        if model > 0:
            for img in images:
                if img.imageType == 1:
                    if config['OS'] == 'linux':
                        if config['THIRD_QUEUE']:
                            keras_queue = 'third_queue'
                            img.queue = 3
                            queue_name = 'C'
                        else:
                            keras_queue = 'first_queue'
                            img.queue = 1
                            queue_name = 'A'
                        q = Queue(keras_queue, connection=Redis())
                        job = q.enqueue_call(init_keras_3D, args=(img.id, img.id, model, refine, 1, [], []), timeout=-1)
                    elif config['OS'] == 'windows':
                        Process(target=init_keras_3D, args=(img.id, img.id, model, refine, 1, [], [])).start()
                        queue_name = 'A'
                    lenq = len(q)
                    img.job_id = job.id
                    if lenq == 0:
                        img.status = 2
                        img.message = 'Processing'
                    else:
                        img.status = 1
                        img.message = 'Queue %s position %s of %s' %(queue_name, lenq, lenq)
                    img.save()
        else:
            request.session['state'] = 'No vaild network selected.'

    # train a neural network
    elif int(action) == 4:

        todo = request.GET.getlist('selected')
        images = Upload.objects.filter(pk__in=todo)

        # get model
        model = 0
        for img in images:
            if img.imageType == 4:
                model = img.id

        # get list of images and labels
        ref_project = 0
        raw_list = ''
        label_list = ''
        for project in range(1, 10):
            raw, label = None, None
            for img in images:
                if img.imageType == 1 and img.project == project:
                    raw = img
                elif img.imageType == 2 and img.project == project:
                    label = img
                elif img.imageType == 3 and img.project == project:
                    label = img
            if raw is not None and label is not None:
                raw_out = raw
                label_out = label
                ref_project = project
                raw_list += raw.pic.path + ':'
                label_list += label.pic.path + ':'

        # train neural network
        if not raw_list:
            request.session['state'] = 'No usable image and label combination selected.'
        else:
            if config['OS'] == 'linux':
                if config['THIRD_QUEUE']:
                    keras_queue = 'third_queue'
                    raw_out.queue = 3
                    queue_name = 'C'
                else:
                    keras_queue = 'first_queue'
                    raw_out.queue = 1
                    queue_name = 'A'
                q = Queue(keras_queue, connection=Redis())
                job = q.enqueue_call(init_keras_3D, args=(raw_out.id, label_out.id, model, 0, 0, raw_list, label_list), timeout=-1)
            elif config['OS'] == 'windows':
                Process(target=init_keras_3D, args=(raw_out.id, label_out.id, model, 0, 0, raw_list, label_list)).start()
                queue_name = 'A'
            lenq = len(q)
            raw_out.job_id = job.id
            if lenq == 0:
                raw_out.status = 2
                raw_out.message = 'Processing'
            else:
                raw_out.status = 1
                raw_out.message = 'Queue %s position %s of %s' %(queue_name, lenq, lenq)
            raw_out.save()

    # active contour
    elif int(action) == 5:
        todo = request.GET.getlist('selected')
        images = Upload.objects.filter(pk__in=todo)
        raw, label = None, None
        for img in images:
            if img.imageType == 1:
                raw = img
            elif img.imageType == 2:
                label = img
            elif img.imageType == 3:
                label = img
        if raw is not None and label is not None:
            from biomedisa_features.active_contour import active_contour
            if config['OS'] == 'linux':
                q = Queue('acwe', connection=Redis())
                job = q.enqueue_call(active_contour, args=(raw.pic.path, label.pic.path, label.id, label.id, raw.id,), timeout=-1)
                lenq = len(q)
                raw.job_id = job.id
                if lenq == 0:
                    raw.status = 2
                    raw.message = 'Processing'
                else:
                    raw.status = 1
                    raw.message = 'Queue D position %s of %s' %(lenq, lenq)
                raw.queue = 4
                raw.save()
            elif config['OS'] == 'windows':
                Process(target=active_contour, args=(raw.pic.path, label.pic.path, label.id, label.id, raw.id)).start()
                raw.status = 2
                raw.message = 'Processing'
                raw.save()
        else:
            request.session['state'] = "No usable image and label combination selected."

    # duplicate file
    elif int(action) == 6:

        todo = request.GET.getlist('selected')
        for id in todo:
            id = int(id)
            stock_to_duplicate = get_object_or_404(Upload, pk=id)
            if stock_to_duplicate.user == request.user:
                CHARACTERS, CODE_SIZE = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz23456789', 7
                newending = id_generator(CODE_SIZE, CHARACTERS)
                if stock_to_duplicate.final:
                    friends = Upload.objects.filter(user=request.user, friend=stock_to_duplicate.friend)
                else:
                    friends = [stock_to_duplicate]

                for k, img in enumerate(friends):

                    # create unique pic path
                    filename, extension = os.path.splitext(img.shortfilename)
                    if extension == '.gz':
                        filename, extension = os.path.splitext(filename)
                        if extension == '.nii':
                            extension = '.nii.gz'
                        elif extension == '.tar':
                            extension = '.tar.gz'
                    addon = ''
                    for feature in ['.cleaned.filled','.smooth.cleaned','.acwe','.smooth',
                                    '.uncertainty','.cleaned','.filled','.8bit','.denoised']:
                        if filename[-len(feature):] == feature:
                            addon = feature
                            filename = filename[:-len(addon)]
                    limit = 100 - len(extension) - len(addon) - 8
                    dir_path = config['PATH_TO_BIOMEDISA'] + '/private_storage/'
                    pic_path = 'images/%s/%s' %(img.user, filename)
                    pic_path = pic_path[:limit] + '_' + newending + addon + extension
                    path_to_data = dir_path + pic_path
                    new_short_name = os.path.basename(pic_path)

                    # create object
                    if img.final:
                        if k == 0:
                            ref_img = Upload.objects.create(pic=pic_path, user=img.user, project=img.project, imageType=img.imageType, shortfilename=new_short_name, final=img.final, active=1)
                            ref_img.friend = ref_img.id
                            ref_img.save()
                        else:
                            Upload.objects.create(pic=pic_path, user=img.user, project=img.project, imageType=img.imageType, shortfilename=new_short_name, final=img.final, friend=ref_img.id)
                    else:
                        Upload.objects.create(pic=pic_path, user=img.user, project=img.project, imageType=img.imageType, shortfilename=new_short_name, final=img.final)

                    # copy data
                    os.link(img.pic.path, path_to_data)

                    # copy slices for visualization
                    path_to_source = img.pic.path.replace('images', 'dataset', 1)
                    path_to_dest = path_to_data.replace('images', 'dataset', 1)
                    if os.path.exists(path_to_source):
                        copytree(path_to_source, path_to_dest, copy_function=os.link)

                    # copy slices for sliceviewer
                    path_to_source = img.pic.path.replace('images', 'sliceviewer', 1)
                    path_to_dest = path_to_data.replace('images', 'sliceviewer', 1)
                    if os.path.exists(path_to_source):
                        copytree(path_to_source, path_to_dest, copy_function=os.link)

                    # copy untared or unzipped data
                    if extension in ['.zip', '.tar', '.tar.gz']:
                        path_to_src = img.pic.path[:-len(extension)]
                        path_to_dir = path_to_data[:-len(extension)]
                        if os.path.exists(path_to_src):
                            copytree(path_to_src, path_to_dir, copy_function=os.link)

    # process image
    elif int(action) in [7,8]:
        started = False
        todo = request.GET.getlist('selected')
        for id in todo:
            id = int(id)
            img = get_object_or_404(Upload, pk=id)
            if img.user == request.user:
                if config['OS'] == 'linux':
                    q = Queue('convert_image', connection=Redis())
                    if int(action) == 7:
                        job = q.enqueue_call(convert_image, args=(img.id,), timeout=-1)
                    elif int(action) == 8:
                        job = q.enqueue_call(smooth_image, args=(img.id,), timeout=-1)
                    lenq = len(q)
                    img.job_id = job.id
                    if lenq == 0:
                        img.status = 2
                        img.message = 'Processing'
                    else:
                        img.status = 1
                        img.message = 'Queue E position %s of %s' %(lenq, lenq)
                    img.queue = 5
                    img.save()
                elif config['OS'] == 'windows':
                    if int(action) == 7:
                        Process(target=convert_image, args=(img.id)).start()
                    elif int(action) == 8:
                        Process(target=smooth_image, args=(img.id)).start()
                    img.status = 2
                    img.message = 'Processing'
                    img.save()
                started = True

    # switch image type
    elif int(action) == 9:
        started = False
        todo = request.GET.getlist('selected')
        for id in todo:
            id = int(id)
            stock_to_change = get_object_or_404(Upload, pk=id)
            if stock_to_change.user == request.user:
                val = int(stock_to_change.imageType)
                if val == 1:
                    stock_to_change.imageType = 2
                    stock_to_change.save()
                elif val == 2:
                    stock_to_change.imageType = 1
                    stock_to_change.save()
                else:
                    request.session['state'] = stock_to_change.shortfilename + " is not an image or label."

# 26. reset
@login_required
def reset(request, id):
    if request.user.is_superuser:
        id = int(id)
        img = get_object_or_404(Upload, pk=id)
        if img.status != 0:
            with open(config['PATH_TO_BIOMEDISA'] + '/log/logfile.txt', 'a') as logfile:
                print('%s reset %s %s' %(time.ctime(), img.user, img.shortfilename), file=logfile)
                img.status = 0
                img.pid = 0
                img.save()
    return redirect(app)

# 27. application
@login_required
def app(request):

    # create profile for superuser
    if request.user.is_superuser:
        try:
            profile = Profile.objects.get(user=request.user)
        except Profile.DoesNotExist:
            profile = Profile(user=request.user)
            profile.save()

            # create user directories
            for directory in ['images', 'dataset', 'sliceviewer']:
                path_to_data = config['PATH_TO_BIOMEDISA'] + '/private_storage/' + directory + '/' + str(profile.user.username)
                if not os.path.isdir(path_to_data):
                    os.makedirs(path_to_data)
                    try:
                        os.chmod(path_to_data, 0o770)
                    except:
                        pass

    # run biomedisa features
    action = request.GET.get('action')
    if action:
        features(request, action)
        return redirect(app)

    # remove symbolic links created for sliceviewer and visualization
    try:
        symlinks = request.session["symlinks"]
        del request.session["symlinks"]
        for symlink in symlinks:
            os.unlink(symlink)
    except:
        pass

    try:
        state = request.session["state"]
    except KeyError:
        state = None

    if request.method == "POST":
        img = UploadForm(request.POST, request.FILES)
        if img.is_valid():
            newimg = Upload(pic=request.FILES['pic'])
            cd = img.cleaned_data
            #image_tmp = Upload.objects.filter(user=request.user, project=cd['project'], imageType=cd['imageType'])
            #if any(image_tmp) and cd['imageType'] in [1,2]:
            #if cd['imageType'] == 1:
            #    request.session['state'] = "Image already exists in project %s" %(cd['project'])
            #elif cd['imageType'] == 2:
            #    request.session['state'] = "Label already exists in project %s" %(cd['project'])
            #return redirect(app)
            #else:
            newimg.project = cd['project']
            newimg.imageType = cd['imageType']
            newimg.user = request.user
            newimg.save()
            newimg.shortfilename = os.path.basename(newimg.pic.path)
            newimg.save()

            # untar or unzip if necessary
            path_to_dir, extension = os.path.splitext(newimg.pic.path)
            if extension == '.gz':
                path_to_dir, extension = os.path.splitext(path_to_dir)
            if extension == '.zip':
                zip_ref = zipfile.ZipFile(newimg.pic.path, 'r')
                zip_ref.extractall(path=path_to_dir)
                zip_ref.close()
            elif extension == '.tar':
                tar = tarfile.open(newimg.pic.path)
                tar.extractall(path=path_to_dir)
                tar.close()

            # create slices
            if newimg.imageType == 1:

                if config['OS'] == 'linux':
                    q = Queue('slices', connection=Redis())
                    job = q.enqueue_call(create_slices, args=(newimg.pic.path, None,), timeout=-1)
                elif config['OS'] == 'windows':
                    Process(target=create_slices, args=(newimg.pic.path, None)).start()

                try:
                    tmp = Upload.objects.get(user=request.user, project=newimg.project, imageType=2)
                    if config['OS'] == 'linux':
                        q = Queue('slices', connection=Redis())
                        job = q.enqueue_call(create_slices, args=(newimg.pic.path, tmp.pic.path,), timeout=-1)
                    elif config['OS'] == 'windows':
                        Process(target=create_slices, args=(newimg.pic.path, tmp.pic.path)).start()
                except Upload.DoesNotExist:
                    pass

            elif newimg.imageType == 2:

                try:
                    tmp = Upload.objects.get(user=request.user, project=newimg.project, imageType=1)
                    path = tmp.pic.path
                except Upload.DoesNotExist:
                    path = None

                if config['OS'] == 'linux':
                    q = Queue('slices', connection=Redis())
                    job = q.enqueue_call(create_slices, args=(path, newimg.pic.path,), timeout=-1)
                elif config['OS'] == 'windows':
                    Process(target=create_slices, args=(path, newimg.pic.path)).start()

            nextType = 2 if newimg.imageType == 1 else 1
            return redirect(reverse('app') + "?project=%s" %(newimg.project) + "&type=%s" %(nextType))

    else:
        img = UploadForm()

    # set initial upload image type
    current_imageType = request.GET.get('type', '')
    img.fields['imageType'].initial = [current_imageType]
    current_project = request.GET.get('project', '')
    img.fields['project'].initial = [current_project]

    # get all images
    images = Upload.objects.filter(user=request.user)
    process_running = 0
    process_list = ""

    # update processing status
    for image in images:

        if image.status > 0:
            process_running = 1
            process_list += ";" + str(image.id) + ":" + str(image.status)

        # one and only one final object is allowed to be active
        if image.final:
            tmp = [x for x in images if image.friend==x.friend]
            if not any(x.active for x in tmp):
                tmp[0].active = 1
                tmp[0].save()

        # update processing status
        if image.status == 1:

            if config['OS'] == 'linux':
                q1 = Queue('first_queue', connection=Redis())
                q2 = Queue('second_queue', connection=Redis())
                q3 = Queue('third_queue', connection=Redis())
                q4 = Queue('acwe', connection=Redis())
                q5 = Queue('convert_image', connection=Redis())
                id_to_check = image.job_id
                if id_to_check in q1.job_ids:
                    i = q1.job_ids.index(id_to_check)
                    image.message = 'Queue A position %s of %s' %(i+1, len(q1))
                elif id_to_check in q2.job_ids:
                    i = q2.job_ids.index(id_to_check)
                    image.message = 'Queue B position %s of %s' %(i+1, len(q2))
                elif id_to_check in q3.job_ids:
                    i = q3.job_ids.index(id_to_check)
                    image.message = 'Queue C position %s of %s' %(i+1, len(q3))
                elif id_to_check in q4.job_ids:
                    i = q4.job_ids.index(id_to_check)
                    image.message = 'Queue D position %s of %s' %(i+1, len(q4))
                elif id_to_check in q5.job_ids:
                    i = q5.job_ids.index(id_to_check)
                    image.message = 'Queue E position %s of %s' %(i+1, len(q5))
                else:
                    image.status = 2
                    image.message = 'Processing'
                image.save()

            elif config['OS'] == 'windows':
                image.status = 2
                image.message = 'Processing'
                image.save()

        # search for failed jobs
        '''elif image.status == 2 and image.pid > 0:
            if config['OS'] == 'linux':
                queue = int(image.queue)
                if queue == 1:
                    host = config['FIRST_QUEUE_HOST']
                elif queue == 2:
                    host = config['SECOND_QUEUE_HOST']
                elif queue == 3:
                    host = config['THIRD_QUEUE_HOST']
                elif queue in [4,5]:
                    host = ''
                q = Queue('check_pid', connection=Redis())
                job = q.enqueue_call(init_check_pid, args=(host, image.pid, image.id), timeout=-1)'''

    # update list of images
    images = Upload.objects.filter(user=request.user)

    # check which projects can be started
    StartProject = np.zeros(9)
    ImageIdRaw = np.zeros(9)
    ImageIdLabel = np.zeros(9)
    max_project = 0

    for k in range(1,10):
        img_obj = Upload.objects.filter(user=request.user, project=k, imageType=1, status=0)
        img_any = Upload.objects.filter(user=request.user, project=k, imageType=1)
        label = Upload.objects.filter(user=request.user, project=k, imageType=2)
        final = Upload.objects.filter(user=request.user, project=k, imageType=3)
        ai = Upload.objects.filter(user=request.user, project=k, imageType=4)
        log = Upload.objects.filter(user=request.user, project=k, log=1)

        if len(img_obj)==1 and len(label)==1 and not final and not ai and not log:
            StartProject[k-1] = 1
            ImageIdRaw[k-1] = img_obj[0].id
            ImageIdLabel[k-1] = label[0].id
        if any([img_any,label,final,ai,log]):
            max_project = k

    looptimes = zip(StartProject, range(1,max_project+1), ImageIdRaw, ImageIdLabel)

    # get storage size of user
    storage_size = request.user.profile.storage_size
    datasize = get_size(config['PATH_TO_BIOMEDISA'] + '/private_storage/images/' + request.user.username)
    storage_full = 1 if datasize > storage_size*10e8 else 0
    datasize *= 10e-10
    if datasize < 1:
        datasize = Decimal(datasize)
        datasize = round(datasize, 3)
    else:
        datasize = int(datasize)

    return render(request, 'app.html', {'state':state, 'loop_times':looptimes, 'form':img, 'images':images,
            'datasize':datasize, 'storage_full':storage_full, 'storage_size':storage_size,
            'process_running':process_running, 'process_list':process_list})

# search for failed jobs (deprecated)
def init_check_pid(host, pid, id):
    cwd = config['PATH_TO_BIOMEDISA']+'/biomedisa_features/'
    if host:
        subprocess.Popen(['ssh',host,'python3',cwd+'check_pid.py',str(pid),str(id)])
    else:
        subprocess.Popen(['python3','check_pid.py',str(pid),str(id)], cwd=cwd)

# search for failed jobs
def init_check_queue(id, processing_queue):
    images = Upload.objects.filter(status=2, queue=processing_queue)
    for img in images:
        if img.id != id:
            img.status = 0
            img.pid = 0
            img.save()
            error_message = 'Something went wrong. Please restart.'
            Upload.objects.create(user=img.user, project=img.project, log=1, imageType=None, shortfilename=error_message)
            path_to_logfile = config['PATH_TO_BIOMEDISA'] + '/log/logfile.txt'
            with open(path_to_logfile, 'a') as logfile:
                print('%s %s %s %s' %(time.ctime(), img.user.username, img.shortfilename, error_message), file=logfile)
            send_error_message(img.user.username, img.shortfilename, error_message)

# 28. constant_time_compare
def constant_time_compare(val1, val2):
    """
    Returns True if the two strings are equal, False otherwise.
    The time taken is independent of the number of characters that match.
    For the sake of simplicity, this function executes in constant time only
    when the two strings have the same length. It short-circuits when they
    have different lengths.
    """
    if len(val1) != len(val2):
        return False
    result = 0
    for x, y in zip(val1, val2):
        result |= ord(x) ^ ord(y)
    return result == 0

# 29. share_data
@login_required
def share_data(request):
    results = {'success':False}
    if request.method == 'GET':
        id = request.GET.get('id')
        list_of_users = request.GET.get('username')
        id = int(id)
        list_of_users = str(list_of_users)

        # share demo file
        demo = request.GET.get('demo')
        if demo:
            id = pk_list[id]

        # get object
        stock_to_share = get_object_or_404(Upload, pk=id)

        # shared by
        shared_id = User.objects.get(username=request.user)
        if shared_id.first_name and shared_id.last_name:
            shared_by = shared_id.first_name + " " + shared_id.last_name
        else:
            shared_by = request.user.username

        if stock_to_share.user == request.user or stock_to_share.user.username == 'demo':
            list_of_users = list_of_users.split(";")
            unknown_users = []
            for new_user_name in list_of_users:

                if User.objects.filter(username=new_user_name).exists():
                    user_id = User.objects.get(username=new_user_name)
                elif User.objects.filter(email=new_user_name).exists():
                    user_id = User.objects.get(email=new_user_name)
                    new_user_name = user_id.username
                else:
                    user_id = None

                if user_id:

                    CHARACTERS, CODE_SIZE = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz23456789', 7
                    newending = id_generator(CODE_SIZE, CHARACTERS)

                    if stock_to_share.final:
                        images = Upload.objects.filter(user=request.user, friend=stock_to_share.friend)
                    else:
                        images = [stock_to_share]

                    for k, img in enumerate(images):

                        # get extension
                        filename, extension = os.path.splitext(img.shortfilename)
                        if extension == '.gz':
                            filename, extension = os.path.splitext(filename)
                            if extension == '.nii':
                                extension = '.nii.gz'
                            elif extension == '.tar':
                                extension = '.tar.gz'

                        # create unique pic path
                        addon = ''
                        for feature in ['.cleaned.filled','.smooth.cleaned','.acwe','.smooth',
                                        '.uncertainty','.cleaned','.filled','.8bit','.denoised']:
                            if filename[-len(feature):] == feature:
                                addon = feature
                                filename = filename[:-len(addon)]

                        dir_path = config['PATH_TO_BIOMEDISA'] + '/private_storage/'
                        pic_path = 'images/%s/%s' %(new_user_name, filename)
                        limit = 100 - len(extension) - len(addon)
                        path_to_data = dir_path + pic_path[:limit] + addon + extension

                        if os.path.exists(path_to_data):
                            limit = 100 - len(extension) - len(addon) - 8
                            pic_path = pic_path[:limit] + '_' + newending + addon + extension
                        else:
                            pic_path = pic_path[:limit] + addon + extension

                        new_short_name = os.path.basename(pic_path)

                        if img.final:
                            if k == 0:
                                name_to_send = new_short_name
                                ref_img = Upload.objects.create(pic=pic_path, user=user_id, project=0, imageType=img.imageType,
                                          shortfilename=new_short_name, final=img.final, shared=1, shared_by=shared_by, shared_path=img.pic.path, active=img.active)
                                ref_img.friend = ref_img.id
                                ref_img.save()
                            else:
                                Upload.objects.create(pic=pic_path, user=user_id, project=0, imageType=img.imageType,
                                shortfilename=new_short_name, final=img.final, shared=1, shared_by=shared_by, shared_path=img.pic.path, friend=ref_img.id, active=img.active)
                        else:
                            name_to_send = new_short_name
                            Upload.objects.create(pic=pic_path, user=user_id, project=0, imageType=img.imageType,
                            shortfilename=new_short_name, final=img.final, shared=1, shared_by=shared_by, shared_path=img.pic.path)

                    if shared_id != user_id:
                        if config['OS'] == 'linux':
                            q = Queue('share_notification', connection=Redis())
                            job = q.enqueue_call(send_share_notify, args=(user_id.username, name_to_send, shared_by,), timeout=-1)
                        elif config['OS'] == 'windows':
                            Process(target=send_share_notify, args=(user_id.username, name_to_send, shared_by)).start()
                else:
                    unknown_users.append(new_user_name)

            if not unknown_users:
                state = "Successfully shared data."
            else:
                state = "Not shared with: " + ' '.join(unknown_users) + " (user does not exist)"

        results = {'success':True, 'msg':state}
    return JsonResponse(results)

# 30. clean_state
@login_required
def clean_state(request, next):
    next = str(next)
    request.session['state'] = None
    if next == 'storage':
        return redirect(storage)
    else:
        return redirect(app)

# 31. create_download_link
@login_required
def create_download_link(request, id):
    id = int(id)
    stock_to_share = get_object_or_404(Upload, pk=id)
    if stock_to_share.user == request.user:
        CHARACTERS, CODE_SIZE = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz23456789', 8
        tmp_pw = id_generator(CODE_SIZE, CHARACTERS)
        stock_to_share.hashed = hash_a_string(tmp_pw)
        link = '%s/download/shared/%s/%s' %(config['SERVER'], id, tmp_pw)
        tmp_pw = id_generator(CODE_SIZE, CHARACTERS)
        stock_to_share.hashed2 = hash_a_string(tmp_pw)
        stock_to_share.save()
        filename = stock_to_share.shortfilename
        return render(request, 'shared_link.html', {'filename':filename, 'pw':tmp_pw, 'link':link})
    else:
        return redirect(app)

# 32. accept_shared_data
@login_required
def accept_shared_data(request):
    results = {'success':False}
    if request.method == 'GET':
        id = request.GET.get('id')
        id = int(id)
        stock_to_share = get_object_or_404(Upload, pk=id)
        if stock_to_share.user == request.user:
            if os.path.isfile(stock_to_share.shared_path):
                CHARACTERS, CODE_SIZE = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz23456789', 7
                newending = id_generator(CODE_SIZE, CHARACTERS)
                if stock_to_share.final:
                    images = Upload.objects.filter(user=request.user, friend=stock_to_share.friend)
                else:
                    images = [stock_to_share]

                for img in images:

                    # get extension
                    filename, extension = os.path.splitext(img.pic.path)
                    if extension == '.gz':
                        filename, extension = os.path.splitext(filename)
                        if extension == '.nii':
                            extension = '.nii.gz'
                        elif extension == '.tar':
                            extension = '.tar.gz'

                    # rename image if pic path already exists
                    if os.path.exists(img.pic.path):

                        addon = ''
                        for feature in ['.cleaned.filled','.smooth.cleaned','.acwe','.smooth',
                                        '.uncertainty','.cleaned','.filled','.8bit','.denoised']:
                            if filename[-len(feature):] == feature:
                                addon = feature
                                filename = filename[:-len(addon)]

                        # max length of filename
                        pic_path = 'images/' + img.user.username + '/' + os.path.basename(filename)
                        limit = 100 - len(extension) - len(addon) - 8
                        pic_path = pic_path[:limit] + '_' + newending + addon + extension

                        # rename django object
                        img.shortfilename = os.path.basename(pic_path)
                        img.pic.name = pic_path

                    # copy file
                    os.link(img.shared_path, img.pic.path)

                    # copy untared or unzipped data
                    if extension in ['.zip', '.tar', '.tar.gz']:
                        path_to_src = img.shared_path[:-len(extension)]
                        path_to_dest = img.pic.path[:-len(extension)]
                        if os.path.exists(path_to_src):
                            copytree(path_to_src, path_to_dest, copy_function=os.link)

                    # copy slices
                    path_to_src = img.shared_path.replace('images', 'dataset', 1)
                    path_to_dest = img.pic.path.replace('images', 'dataset', 1)
                    if os.path.exists(path_to_src):
                        copytree(path_to_src, path_to_dest, copy_function=os.link)

                    # copy dataset
                    path_to_src = img.shared_path.replace('images', 'sliceviewer', 1)
                    path_to_dest = img.pic.path.replace('images', 'sliceviewer', 1)
                    if os.path.exists(path_to_src):
                        copytree(path_to_src, path_to_dest, copy_function=os.link)

                    # close sharing
                    img.shared = 0
                    img.save()

                results = {'success':True}
            else:
                state = 'Data has already been removed.'
                results = {'success':True,'msg':state}
    return JsonResponse(results)

# 33. download_shared_data
def download_shared_data(request, id, pw):
    id = int(id)
    pw = str(pw)
    stock_to_share = get_object_or_404(Upload, pk=id)
    hashed_pw = hash_a_string(pw)
    if constant_time_compare(stock_to_share.hashed, hashed_pw):
        c = {}
        c.update(csrf(request))
        state = 'Please enter the password...'
        password = ''
        stock_to_share = get_object_or_404(Upload, pk=id)
        filename = stock_to_share.shortfilename
        shared_by = stock_to_share.user
        size = stock_to_share.size
        if request.POST:
            password = request.POST.get('password')
            hashed = hash_a_string(password)
            stock_to_download = get_object_or_404(Upload, pk=id)
            if constant_time_compare(stock_to_download.hashed2, hashed):
                filename = stock_to_download.shortfilename
                path_to_file = stock_to_download.pic.path
                wrapper = FileWrapper(open(path_to_file, 'rb'))
                imgsize = os.path.getsize(path_to_file)
                if imgsize < 5000000000:
                    response = HttpResponse(wrapper, content_type='%s' %(filename))
                else:
                    response = StreamingHttpResponse(wrapper, content_type='%s' %(filename))
                response['Content-Disposition'] = 'attachment; filename="%s"' %(filename)
                response['Content-Length'] = imgsize
                return response
            else:
                state = 'The password was incorrect.'
        return render(request, ('download_shared_data.html', c), {'state': state, 'filename': filename, 'shared_by': shared_by, 'size': size})
    else:
        return redirect(index)

# 34. download
@login_required
def download(request, id):
    id = int(id)
    stock_to_download = get_object_or_404(Upload, pk=id)
    if stock_to_download.user == request.user:
        filename = stock_to_download.shortfilename
        path_to_file = stock_to_download.pic.path
        wrapper = FileWrapper(open(path_to_file, 'rb'))
        imgsize = os.path.getsize(path_to_file)
        if imgsize < 5000000000:
            response = HttpResponse(wrapper, content_type='%s' %(filename))
        else:
            response = StreamingHttpResponse(wrapper, content_type='%s' %(filename))
        response['Content-Disposition'] = 'attachment; filename="%s"' %(filename)
        response['Content-Length'] = imgsize
        return response

# 35. gallery
def gallery(request):
    return render(request, 'gallery.html', {'k': request.session.get('k')})

# 36. run_demo
def run_demo(request):
    request.session['k'] += 1
    return redirect(gallery)

# 37. delete_demo
def delete_demo(request):
    request.session['k'] = 0
    return redirect(gallery)

# 38. download_demo
def download_demo(request, id):
    int_id = int(id)
    file_names = ["screw_joint.tif", "labels.screw_joint.tif", "final.screw_joint.tif", \
                 "trigonopterus.tif", "labels.trigonopterus_smart.am", "final.trigonopterus_smart.am", \
                 "tumor.tif", "labels.tumor.tif", "final.tumor.itf", \
                 "NMB_F2875.tif", "labels.NMB_F2875.tif", "final.NMB_F2875.tif", \
                 "NRM-PZ_Ar65720.tif", "labels.NRM-PZ_Ar65720.am", "final.NRM-PZ_Ar65720.am", \
                 "training_heart.tar", "training_heart_labels.tar", "heart.h5", \
                 "testing_axial_crop_pat13.nii.gz", "final.testing_axial_crop_pat13.tif", \
                 'wasp_from_amber.tif', 'labels.wasp_from_amber.am', 'final.wasp_from_amber.am', \
                 'cockroach.tif', 'labels.cockroach.am', 'final.cockroach.am', \
                 'theropod_claw.tif', 'labels.theropod_claw.tif', 'final.theropod_claw.tif', \
                 'bull_ant_queen.am', 'labels.bull_ant_queen_head.am', 'final.bull_ant_queen_head.am', \
                 'mouse_molar_teeth.tar', 'labels.mouse_molar_teeth.tar', 'final.mouse_molar_teeth.tar']
    if 0 <= int_id < len(pk_list):
        filename = file_names[int_id]
        stock_to_download = get_object_or_404(Upload, pk=pk_list[int_id])
        path_to_file = stock_to_download.pic.path
        wrapper = FileWrapper(open(path_to_file, 'rb'))
        imgsize = os.path.getsize(path_to_file)
        if imgsize < 5000000000:
            response = HttpResponse(wrapper, content_type='%s' %(filename))
        else:
            response = StreamingHttpResponse(wrapper, content_type='%s' %(filename))
        response['Content-Disposition'] = 'attachment; filename="%s"' %(filename)
        response['Content-Length'] = imgsize
        return response

# 39. visualization_demo
def visualization_demo(request, id):
    int_id = int(id)
    if 0 <= int_id < len(pk_list):
        stock_to_visualize = get_object_or_404(Upload, pk=pk_list[int_id])
        filename = stock_to_visualize.shortfilename
        noslices = len(os.listdir(config['PATH_TO_BIOMEDISA'] + '/media/dataset/demo/%s' %(filename)))
        path_to_slicemaps = ['/media/dataset/demo/%s/slicemap_%s.png' %(filename, k) for k in range(noslices)]
        path_to_slicemaps = json.dumps(path_to_slicemaps)
        return render(request, 'visualization.html', {'path_to_slicemaps':path_to_slicemaps})

# 40. sliceviewer_demo
def sliceviewer_demo(request, id):
    int_id = int(id)
    if 0 <= int_id < len(pk_list):
        stock_to_show = get_object_or_404(Upload, pk=pk_list[int_id])
        filename = stock_to_show.shortfilename
        path_to_slices = "/media/sliceviewer/demo/%s" %(filename)
        full_path = config['PATH_TO_BIOMEDISA'] + path_to_slices
        nos = len(os.listdir(full_path)) - 1
        path_to_slices += '/'
        im = Image.open(full_path + '/0.png')
        imshape = np.asarray(im).shape
        return render(request, 'sliceviewer.html', {'path_to_slices':path_to_slices, 'nos':nos, 'imshape_x':imshape[1], 'imshape_y':imshape[0]})

# 41. delete
@login_required
def delete(request):
    results = {'success':False}
    if request.method == 'GET':
        id = int(request.GET.get('id'))
        try:
            stock_to_delete = get_object_or_404(Upload, pk=id)
            if stock_to_delete.user == request.user and stock_to_delete.status==0:
                if stock_to_delete.final:
                    images = Upload.objects.filter(user=request.user, friend=stock_to_delete.friend)
                    for img in images:
                        img.delete()
                else:
                    stock_to_delete.delete()
            results = {'success':True}
        except:
            pass
    return JsonResponse(results)

# 42. init_random_walk
def init_random_walk(image, label, second_queue):

    # get objects
    try:
        image = Upload.objects.get(pk=image)
        label = Upload.objects.get(pk=label)
    except Upload.DoesNotExist:
        image.status = 0
        image.pid = 0
        image.save()
        message = 'Files have been removed.'
        Upload.objects.create(user=image.user, project=image.project, log=1, imageType=None, shortfilename=message)

    # check if aborted
    if image.status > 0:

        # set status to processing
        if image.status == 1:
            image.status = 2
            image.message = 'Processing'
            image.save()

        # change working directory
        cwd = config['PATH_TO_BIOMEDISA'] + '/biomedisa_features/random_walk/'
        workers_host = config['PATH_TO_BIOMEDISA'] + '/log/workers_host'

        # get configuration
        if second_queue:
            ngpus = str(config['SECOND_QUEUE_NGPUS'])
            host = config['SECOND_QUEUE_HOST']
            cluster = config['SECOND_QUEUE_CLUSTER']
        else:
            ngpus = str(config['FIRST_QUEUE_NGPUS'])
            host = config['FIRST_QUEUE_HOST']
            cluster = config['FIRST_QUEUE_CLUSTER']

        # run random walks
        if config['OS'] == 'linux':

            if host and cluster:
                p = subprocess.Popen(['ssh', host, 'mpiexec', '-np', ngpus, '--hostfile', workers_host, 'python3', cwd+'rw_main.py', str(image.id), str(label.id)])
            elif host:
                p = subprocess.Popen(['ssh', host, 'mpiexec', '-np', ngpus, 'python3', cwd+'rw_main.py', str(image.id), str(label.id)])
            elif cluster:
                p = subprocess.Popen(['mpiexec', '-np', ngpus, '--hostfile', workers_host, 'python3', 'rw_main.py', str(image.id), str(label.id)], cwd=cwd)
            else:
                p = subprocess.Popen(['mpiexec', '-np', ngpus, 'python3', 'rw_main.py', str(image.id), str(label.id)], cwd=cwd)

        elif config['OS'] == 'windows':
            p = subprocess.Popen(['mpiexec', '-np', ngpus, 'python', '-u', 'rw_main.py', str(image.id), str(label.id)], cwd=cwd, stdout=subprocess.PIPE)

        # print output
        if config['OS'] == 'windows':
            for line in iter(p.stdout.readline, b''):
                line = str(line,'utf-8')
                print(line.rstrip())
            p.stdout.close()

        # wait for process terminating
        p.wait()

# 43. run random walks
@login_required
def run(request):
    results = {'success':False}
    if request.method == 'GET':
        raw = int(request.GET.get('raw'))
        label = int(request.GET.get('label'))
        try:
            raw = get_object_or_404(Upload, pk=raw)
            label = get_object_or_404(Upload, pk=label)

            if raw.user == request.user and label.user == request.user:

                if config['OS'] == 'linux':

                    # two processing queues
                    if config['SECOND_QUEUE']:
                        q1 = Queue('first_queue', connection=Redis())
                        q2 = Queue('second_queue', connection=Redis())
                        lenq1 = len(q1)
                        lenq2 = len(q2)

                        if lenq1 > lenq2:
                            second_queue = True
                            job = q2.enqueue_call(init_random_walk, args=(raw.id, label.id, second_queue), timeout=-1)
                            lenq = len(q2)
                        else:
                            second_queue = False
                            job = q1.enqueue_call(init_random_walk, args=(raw.id, label.id, second_queue), timeout=-1)
                            lenq = len(q1)
                            if lenq > 0:
                                job.delete()
                                second_queue = True
                                job = q2.enqueue_call(init_random_walk, args=(raw.id, label.id, second_queue), timeout=-1)
                                lenq = len(q2)
                                if lenq > 0:
                                    job.delete()
                                    second_queue = False
                                    job = q1.enqueue_call(init_random_walk, args=(raw.id, label.id, second_queue), timeout=-1)
                                    lenq = len(q1)

                        raw.job_id = job.id
                        if lenq == 0:
                            raw.status = 2
                            raw.message = 'Processing'
                        else:
                            raw.status = 1
                            cluster = 'B' if second_queue else 'A'
                            raw.message = 'Queue %s position %s of %s' %(cluster, lenq, lenq)
                        if second_queue:
                            raw.queue = 2
                        else:
                            raw.queue = 1
                        raw.save()

                    # single processing queue
                    else:
                        q1 = Queue('first_queue', connection=Redis())
                        job = q1.enqueue_call(init_random_walk, args=(raw.id, label.id, False), timeout=-1)
                        lenq = len(q1)
                        raw.job_id = job.id
                        if lenq == 0:
                            raw.status = 2
                            raw.message = 'Processing'
                        else:
                            raw.status = 1
                            raw.message = 'Queue A position %s of %s' %(lenq, lenq)
                        raw.queue = 1
                        raw.save()

                    # search for failed jobs in queue
                    q = Queue('check_pid', connection=Redis())
                    job = q.enqueue_call(init_check_queue, args=(raw.id, raw.queue,), timeout=-1)

                elif config['OS'] == 'windows':
                    Process(target=init_random_walk, args=(raw.id, label.id, False)).start()
                    raw.status = 2
                    raw.message = 'Processing'
                    raw.queue = 1
                    raw.save()

            results = {'success':True}
        except:
            pass
    return JsonResponse(results)

# 44. stop running job
def stop_running_job(id, queue):
    id = int(id)
    queue = int(queue)

    # get configuration
    if queue == 1:
        host = config['FIRST_QUEUE_HOST']
    elif queue == 2:
        host = config['SECOND_QUEUE_HOST']
    elif queue == 3:
        host = config['THIRD_QUEUE_HOST']
    elif queue in [4,5]:
        host = ''

    # kill process
    try:
        if config['OS'] == 'linux':
            if host:
                subprocess.Popen(['ssh', host, 'kill', str(id)])
            else:
                subprocess.Popen(['kill', str(id)])
        elif config['OS'] == 'windows':
            subprocess.Popen(['Taskkill', '/PID', str(id), '/F'])
    except:
        pass

# 45. remove_from_queue or kill process
@login_required
def remove_from_queue(request):
    results = {'success':False}
    if request.method == 'GET':
        image_to_stop = int(request.GET.get('id'))
        try:
            image_to_stop = get_object_or_404(Upload, pk=image_to_stop)
            if image_to_stop.user == request.user:

                # remove from queue
                if config['OS'] == 'linux':
                    q1 = Queue('first_queue', connection=Redis())
                    q2 = Queue('second_queue', connection=Redis())
                    q3 = Queue('third_queue', connection=Redis())
                    q4 = Queue('acwe', connection=Redis())
                    q5 = Queue('convert_image', connection=Redis())
                    id_to_check = image_to_stop.job_id
                    if id_to_check in q1.job_ids:
                        job = q1.fetch_job(id_to_check)
                        job.delete()
                    elif id_to_check in q2.job_ids:
                        job = q2.fetch_job(id_to_check)
                        job.delete()
                    elif id_to_check in q3.job_ids:
                        job = q3.fetch_job(id_to_check)
                        job.delete()
                    elif id_to_check in q4.job_ids:
                        job = q4.fetch_job(id_to_check)
                        job.delete()
                    elif id_to_check in q5.job_ids:
                        job = q5.fetch_job(id_to_check)
                        job.delete()

                # kill running process
                if image_to_stop.status == 2 and image_to_stop.pid > 0:
                    if config['OS'] == 'linux':
                        q = Queue('stop_job', connection=Redis())
                        job = q.enqueue_call(stop_running_job, args=(image_to_stop.pid, image_to_stop.queue), timeout=-1)
                    elif config['OS'] == 'windows':
                        stop_running_job(image_to_stop.pid, image_to_stop.queue)
                    image_to_stop.pid = 0
                    image_to_stop.save()

                # reset image
                image_to_stop.status = 0
                image_to_stop.save()

                # reset all images in project
                #images = Upload.objects.filter(user=request.user, project=image_to_stop.project)
                #for img in images:
                #    img.status = 0
                #    img.save()

            results = {'success':True}

        except:
            pass
    return JsonResponse(results)

# 46. delete account
@login_required
def delete_account(request):
    content = {}
    try:
        user_id = User.objects.get(username=request.user)
        profile = Profile.objects.get(user=request.user)
        images = Upload.objects.filter(user=request.user)
        for img in images:
            img.delete()
        # delete user directories
        for directory in ['images', 'dataset', 'sliceviewer']:
            path_to_data = config['PATH_TO_BIOMEDISA'] + '/private_storage/' + directory + '/' + request.user.username
            shutil.rmtree(path_to_data)
        profile.delete()
        user_id.delete()
        logout(request)
        content['msg'] = 'Your account and all your data have been deleted successfully!'
    except User.DoesNotExist:
        content['msg'] = 'User does not exist.'
    except Exception as e:
        content['msg'] = e.message
    return render(request, 'contact.html', content)

# 47. send notification
def send_notification(username, image_name, time_str, server_name, train=False):

    if train:
        info = 'trained a neural network for'
    else:
        info = 'finished the segmentation of'

    user = User.objects.get(username=username)
    if user.first_name:
        username = user.first_name

    recipients = [config['EMAIL']]
    if user.profile.notification and user.email and user.email not in recipients:
        recipients.append(user.email)

    for recipient in recipients:

        # prepare data package
        datas={}
        datas['email'] = recipient
        datas['email_path'] = '/ProcessCompleted.txt'
        datas['email_subject'] = 'Segmentation successfully finished'
        datas['context'] = Context({'host':config['SERVER'], 'ctime':time_str, 'server_name': server_name,
                                    'username':username, 'image_name':image_name, 'info':info})

        # send notification
        sendEmail(datas)

# 48. start notification
def send_start_notification(image):

    # prepare data package
    datas={}
    datas['email'] = config['EMAIL']
    datas['email_path'] = '/MessageEmail.txt'
    datas['email_subject'] = 'Process was started'
    msg = 'The user %s started a process for %s on %s.' %(image.user.username, image.shortfilename, config['SERVER_ALIAS'])
    datas['context'] = Context({'msg':msg})

    # send notification
    sendEmail(datas)

# 49. error message
def send_error_message(username, image_name, error_msg):

    user = User.objects.get(username=username)
    if user.first_name:
        username = user.first_name

    recipients = [config['EMAIL']]
    if user.profile.notification and user.email and user.email not in recipients:
        recipients.append(user.email)

    for recipient in recipients:

        # prepare data package
        datas={}
        datas['email'] = recipient
        datas['email_path'] = '/ErrorEmail.txt'
        datas['email_subject'] = 'Segmentation error'
        datas['context'] = Context({'host':config['SERVER'], 'error_msg':error_msg, 'username':username, 'image_name':image_name})

        # send notification
        sendEmail(datas)

# 50. send_share_notify
def send_share_notify(username, image_name, shared_by):

    user = User.objects.get(username=username)
    if user.first_name:
        username = user.first_name

    if user.profile.notification and user.email:

        # prepare data package
        datas={}
        datas['email'] = user.email
        datas['email_path'] = '/ShareEmail.txt'
        datas['email_subject'] = shared_by + ' wants to share data with you'
        datas['context'] = Context({'host':config['SERVER'], 'shared_by':shared_by, 'username':username, 'image_name':image_name})

        # send notification
        sendEmail(datas)

# 51. status function
@login_required
def status(request):
    results = {'success':False}
    if request.method == 'GET':
        inp = request.GET.get('ids')
        inp = inp.strip(";").split(";")

        status = {}
        for i in inp:
            x = i.split(":")
            status[int(x[0])] = int(x[1]) # image status

        reload = False
        images = Upload.objects.filter(pk__in=status.keys())
        for im in images:
            if im.status != status[im.id]:
                reload = True
                break

        results = {'success':True, 'reload':reload}
    return JsonResponse(results)

# 52. dummy function
@login_required
def dummy(request):
    # do nothing
    results = {'success':True}
    return JsonResponse(results)
