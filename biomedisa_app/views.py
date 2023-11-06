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

from __future__ import unicode_literals

import django
django.setup()
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, StreamingHttpResponse, JsonResponse, HttpResponseRedirect
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
from django.db.models import Q

from biomedisa_app.models import (UploadForm, Upload, StorageForm, Profile,
    UserForm, SettingsForm, SettingsPredictionForm, CustomUserCreationForm,
    Repository, Specimen, SpecimenForm, TomographicData, TomographicDataForm,
    ProcessedData)
from biomedisa_features.create_slices import create_slices
from biomedisa_features.biomedisa_helper import (load_data, save_data, img_to_uint8, id_generator,
    convert_image, smooth_image, convert_to_stl, unique_file_path, _get_platform)
from django.utils.crypto import get_random_string
from biomedisa_app.config import config
from biomedisa.settings import BASE_DIR, WWW_DATA_ROOT
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
import shutil, wget
import subprocess
import glob

from redis import Redis
from rq import Queue, Worker, get_current_job

class Biomedisa(object):
     pass

# 01. paraview
def paraview(request):
    return render(request, 'paraview.html')

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

@login_required
def repository(request, id=1):

    try:
        state = request.session["state"]
    except KeyError:
        state = None

    # get repository
    id = int(id)
    repository = get_object_or_404(Repository, pk=id)

    # search for specimen containing query
    if request.user in repository.users.all():
        query = request.GET.get('search')
        if query:
            specimens = Specimen.objects.filter(Q(internal_id__icontains=query) | Q(subfamily__icontains=query) | Q(genus__icontains=query)\
             | Q(species__icontains=query) | Q(caste__icontains=query), repository=repository, sketchfab=None)
            specimens_with_model = Specimen.objects.filter(Q(internal_id__icontains=query) | Q(subfamily__icontains=query) | Q(genus__icontains=query)\
             | Q(species__icontains=query) | Q(caste__icontains=query), repository=repository).exclude(sketchfab=None)
        else:
            specimens = Specimen.objects.filter(repository=repository, sketchfab=None)
            specimens_with_model = Specimen.objects.filter(repository=repository).exclude(sketchfab=None)
        all_specimens = Specimen.objects.filter(repository=repository)
        return render(request, 'repository.html', {'state':state, 'specimens':specimens, 'repository_alias':repository.repository_alias,
                    'specimens_with_model':specimens_with_model, 'all_specimens':all_specimens, 'featured_img':repository.featured_img,
                    'featured_img_width':repository.featured_img_width, 'featured_img_height':repository.featured_img_height})

@login_required
def share_repository(request):
    results = {'success':False}
    if request.method == 'GET':
        id = int(request.GET.get('id'))
        username = str(request.GET.get('username'))
        repository = get_object_or_404(Repository, pk=id)
        if request.user in repository.users.all():
            if User.objects.filter(username=username).exists():
                user_to_add = User.objects.get(username=username)
                repository.users.add(user_to_add)
                repository.save()
                results = {'success':True, 'msg':'Repository shared successfully.'}
                q = Queue('share_notification', connection=Redis())
                job = q.enqueue_call(repository_share_notify, args=(username, repository.repository_alias, request.user.username,), timeout=-1)
            else:
                results = {'success':True, 'msg':'User does not exist.'}
    return JsonResponse(results)

def repository_share_notify(username, repository, shared_by):
    user = User.objects.get(username=username)
    if user.first_name:
        username = user.first_name
    if user.profile.notification and user.email:
        # prepare data package
        datas={}
        datas['email'] = user.email
        datas['email_path'] = '/ShareRepository.txt'
        datas['email_subject'] = shared_by + ' shared a repository with you'
        datas['context'] = Context({'host':config['SERVER'], 'shared_by':shared_by, 'username':username, 'repository':repository})
        # send notification
        sendEmail(datas)

@login_required
def unsubscribe_from_repository(request, id):
    results = {'success':False}
    repository = get_object_or_404(Repository, pk=id)
    user_to_remove = User.objects.get(username=request.user)
    if request.user in repository.users.all():
        repository.users.remove(user_to_remove)
        repository.save()
        results = {'success':True}
    return JsonResponse(results)

@login_required
def specimen_info(request, id):
    id = int(id)
    specimen = get_object_or_404(Specimen, pk=id)
    if request.user in specimen.repository.users.all():
        # initialization
        initial = {}
        specimen_form = SpecimenForm()
        for key in specimen_form.fields.keys():
            initial[key] = specimen.__dict__[key]
        # get data
        if request.method == 'POST':
            data = SpecimenForm(request.POST)
            if data.is_valid():
                cd = data.cleaned_data
                if cd != initial:
                    if cd['sketchfab'] == '':
                        cd['sketchfab'] = None
                    for key in cd.keys():
                        specimen.__dict__[key] = cd[key]
                    specimen.save()
                    messages.success(request, 'Information updated successfully.')
                return redirect(specimen_info, id)
        else:
            # preview stitched tomographic data
            nos = 0
            imshape = (0,0)
            path_to_slices = None
            processed_data = ProcessedData.objects.filter(specimen=specimen, imageType=1)
            if ProcessedData.objects.filter(specimen=specimen, imageType=1).exists():
                processed_data = ProcessedData.objects.filter(specimen=specimen, imageType=1)[0]
                path_to_slices = os.path.splitext('/media/antscan/' + processed_data.pic.name)[0]
                full_path = BASE_DIR + path_to_slices
                if os.path.exists(full_path):
                    nos = len(os.listdir(full_path)) - 1
                    path_to_slices += '/'
                    im = Image.open(full_path + '/0.png')
                    imshape = np.asarray(im).shape
            # form
            specimen_form = SpecimenForm(initial=initial)
            name = specimen.internal_id if not any([specimen.name_recommended, specimen.subfamily, specimen.caste, specimen.specimen_code]) else "{name_recommended} | {subfamily} | {caste} | {specimen_code}".format(name_recommended=specimen.name_recommended, subfamily=specimen.subfamily, caste=specimen.caste, specimen_code=specimen.specimen_code)
            tomographic_data = TomographicData.objects.filter(specimen=specimen)
            processed_data = ProcessedData.objects.filter(specimen=specimen)
            sketchfab_id = specimen.sketchfab
            return render(request, 'specimen_info.html', {'specimen_form':specimen_form,'tomographic_data':tomographic_data,
                                                          'processed_data':processed_data,'name':name,'specimen':specimen,
                                                          'path_to_slices':path_to_slices,'nos':nos, 'imshape_x':imshape[1], 'imshape_y':imshape[0]
                                                         })

@login_required
def tomographic_info(request, id):
    id = int(id)
    tomographic_data = get_object_or_404(TomographicData, pk=id)
    if request.user in tomographic_data.specimen.repository.users.all():
        # initialization
        initial = {}
        tomographic_form = TomographicDataForm()
        for key in tomographic_form.fields.keys():
            initial[key] = tomographic_data.__dict__[key]
        if request.method == 'POST':
            data = TomographicDataForm(request.POST)
            if data.is_valid():
                cd = data.cleaned_data
                if cd != initial:
                    for key in cd.keys():
                        tomographic_data.__dict__[key] = cd[key]
                    tomographic_data.save()
                    messages.success(request, 'Information updated successfully.')
                return redirect(tomographic_info, id)
        else:
            # preview tomographic data
            path_to_slices = '/media/' + os.path.dirname(tomographic_data.pic.name) + '/slices'
            full_path = BASE_DIR + path_to_slices
            if os.path.exists(full_path):
                nos = len(os.listdir(full_path)) - 1
                path_to_slices += '/'
                im = Image.open(full_path + '/0.png')
                imshape = np.asarray(im).shape
                imshape_x = 400
                imshape_y = int(imshape[0]/imshape[1]*400)
                imshape = (imshape_y,imshape_x)
            else:
                nos = 0
                imshape = (0,0)
            # tomographic form
            tomographic_form = TomographicDataForm(initial=initial)
            return render(request, 'tomographic_info.html', {'tomographic_form':tomographic_form,'name':tomographic_data.pic.name,
                                                             'related_specimen':tomographic_data.specimen.id, 'path_to_slices':path_to_slices,
                                                             'nos':nos, 'imshape_x':imshape[1], 'imshape_y':imshape[0]})

@login_required
def sliceviewer_repository(request):
    if request.method == 'GET':
        id = int(request.GET.get('id'))
        obj = str(request.GET.get('object'))[:11]
        if obj == 'tomographic':
            tomographic_data = get_object_or_404(TomographicData, pk=id)
        elif obj == 'processed':
            tomographic_data = get_object_or_404(ProcessedData, pk=id)
        elif obj == 'specimen':
            specimen = get_object_or_404(Specimen, pk=id)
            tomographic_data = ProcessedData.objects.filter(specimen=specimen, imageType=1)[0]
        if request.user in tomographic_data.specimen.repository.users.all():
            if obj == 'processed' or obj == 'specimen':
                path_to_slices = '/media/antscan/' + tomographic_data.pic.name.replace('.tif','')
            else:
                path_to_slices = '/media/' + os.path.dirname(tomographic_data.pic.name) + '/slices'
            full_path = BASE_DIR + path_to_slices
            nos = len(os.listdir(full_path)) - 1
            path_to_slices += '/'
            im = Image.open(full_path + '/0.png')
            imshape = np.asarray(im).shape
            return render(request, 'sliceviewer.html', {'path_to_slices':path_to_slices, 'nos':nos, 'imshape_x':imshape[1], 'imshape_y':imshape[0]})

@login_required
def visualization_repository(request):
    if request.method == 'GET':
        id = int(request.GET.get('id'))
        obj = str(request.GET.get('object'))[:11]
        if obj == 'processed':
            specimen = get_object_or_404(ProcessedData, pk=id).specimen
        elif obj == 'specimen':
            specimen = get_object_or_404(Specimen, pk=id)
        if request.user in specimen.repository.users.all():
            path_to_link = '/media/antscan/2020_12_antscan/' + specimen.internal_id + '.stl'
            name = specimen.internal_id + '.stl'
            url = config['SERVER'] + path_to_link
            URL = config['SERVER'] + "/paraview/?name=["+name+"]&url=["+url+"]"
            return HttpResponseRedirect(URL)

@login_required
def download_repository(request):
    if request.method == 'GET':
        id = int(request.GET.get('id'))
        obj = str(request.GET.get('object'))[:11]
        if obj == 'tomographic':
            tomographic_data = get_object_or_404(TomographicData, pk=id)
        elif obj == 'processed':
            tomographic_data = get_object_or_404(ProcessedData, pk=id)
        if request.user in tomographic_data.specimen.repository.users.all():
            filename = tomographic_data.pic.name
            path_to_file = tomographic_data.pic.path
            if obj == 'processed':
                path_to_file = path_to_file.replace('media','media/antscan')
            wrapper = FileWrapper(open(path_to_file, 'rb'))
            imgsize = os.path.getsize(path_to_file)
            if imgsize < 5000000000:
                response = HttpResponse(wrapper, content_type=filename)
            else:
                response = StreamingHttpResponse(wrapper, content_type=filename)
            response['Content-Disposition'] = 'attachment; filename="%s"' %(filename)
            response['Content-Length'] = imgsize
            return response

@login_required
def share_repository_data(request):
    results = {'success':False}
    if request.method == 'GET':
        id = int(request.GET.get('id'))

        tomographic_data = get_object_or_404(TomographicData, pk=id)
        if request.user in tomographic_data.specimen.repository.users.all():

            # new file path
            shortfilename = tomographic_data.pic.name.replace('/','_')
            pic_path = 'images/' + request.user.username + '/' + shortfilename

            # rename image if path already exists
            if os.path.exists(WWW_DATA_ROOT + '/' + pic_path):
                path_to_data = unique_file_path(pic_path, request.user.username, WWW_DATA_ROOT+'/')
                pic_path = 'images/' + request.user.username + '/' + os.path.basename(path_to_data)
                shortfilename = os.path.basename(path_to_data)

            # create object
            img = Upload.objects.create(pic=pic_path, user=request.user, project=0, shortfilename=shortfilename)

            # copy file
            shutil.copy2(tomographic_data.pic.path, img.pic.path)

            # copy slices
            path_to_src = os.path.dirname(tomographic_data.pic.path) + '/slices'
            path_to_dest = img.pic.path.replace('images', 'sliceviewer', 1)
            if os.path.exists(path_to_src):
                copytree(path_to_src, path_to_dest)

            results = {'success':True, 'msg':'Successfully shared data.'}
    return JsonResponse(results)

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
    f = open(BASE_DIR + '/biomedisa_app/messages' + datas['email_path'], 'r')
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
                for directory in ['images', 'sliceviewer']:
                    path_to_data = WWW_DATA_ROOT + '/' + directory + '/' + meta['username']
                    os.makedirs(path_to_data)
                    try:
                        os.chmod(path_to_data, 0o770)
                    except:
                        pass
                messages.success(request, 'Account created successfully.')

            # write in biomedisa/log/applications.txt
            afile = open(BASE_DIR + '/log/applications.txt', 'a')
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
        num1, num2 = np.random.randint(0, high=31, size=2, dtype=int)
        f = CustomUserCreationForm()
        f.fields["numeric1"].initial = str(num1)
        f.fields["numeric2"].initial = str(num2)
        f.fields["verification"].initial = 'Please enter the sum of '+str(num1)+' and '+str(num2)

    return render(request, 'register.html', {'form':f})

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
            for directory in ['images', 'sliceviewer']:
                path_to_data = WWW_DATA_ROOT + '/' + directory + '/' + str(profile.user.username)
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
            if val in [2,3,7,8,10]:
                request.session['state'] = "Still processing. Please wait."
            elif val in [4,5]:
                request.session['state'] = "Result not available (not available for AI segmentation or GPUs out of memory)"
            elif val==6:
                if any(Upload.objects.filter(user=request.user, friend=stock_to_change.friend, final=5)):
                    request.session['state'] = "Still processing. Please wait."
                else:
                    request.session['state'] = "Result not available (not available for AI segmentation or GPUs out of memory)"
            elif val==9:
                request.session['state'] = "Result not available (only available for AI segmentation with auto-cropping)"
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
        src = stock_to_show.pic.path.replace("images", "sliceviewer", 1)

        if os.path.isdir(src):
            # link data from stoarge to media
            prefix = generate_activation_key()
            path_to_slices = '/media/' + prefix
            dest = BASE_DIR + path_to_slices
            os.symlink(src, dest)
            path_to_slices += '/'

            # create symlinks wich are removed when "app" is called or user loggs out
            try:
                symlinks = request.session["symlinks"]
                symlinks.append(dest)
                request.session["symlinks"] = symlinks
            except:
                request.session["symlinks"] = [dest]

            nos = len(os.listdir(dest)) - 1
            im = Image.open(dest + '/0.png')
            imshape = np.asarray(im).shape
            return render(request, 'sliceviewer.html', {'path_to_slices':path_to_slices, 'nos':nos, 'imshape_x':imshape[1], 'imshape_y':imshape[0]})

        else:
            # create slices
            q = Queue('slices', connection=Redis())
            if stock_to_show.imageType == 1:
                job = q.enqueue_call(create_slices, args=(stock_to_show.pic.path, None,), timeout=-1)
            elif stock_to_show.imageType in [2,3]:
                images = Upload.objects.filter(user=request.user, project=stock_to_show.project, imageType=1)
                if len(images)>0:
                    job = q.enqueue_call(create_slices, args=(images[0].pic.path, stock_to_show.pic.path,), timeout=-1)

            request.session['state'] = 'The slice preview is calculated. Please wait.'
            next = request.GET.get('next', '')
            next = str(next)
            if next == 'storage':
                return redirect(storage)
            else:
                return redirect(app)

# 18. visualization
@login_required
def visualization(request):
    if request.method == 'GET':

        # download paraview
        if not os.path.exists(BASE_DIR + '/biomedisa_app/paraview'):
            wget.download('https://biomedisa.org/media/paraview.zip', out=BASE_DIR + '/biomedisa_app/paraview.zip')
            zip_ref = zipfile.ZipFile(BASE_DIR + '/biomedisa_app/paraview.zip', 'r')
            zip_ref.extractall(path=BASE_DIR + '/biomedisa_app')
            zip_ref.close()

        ids = request.GET.get('selected','')
        ids = ids.split(',')
        name,url="",""
        amira_not_supported = False
        for id in ids:
            id = int(id)
            stock_to_show = get_object_or_404(Upload, pk=id)
            if stock_to_show.user == request.user:

                prefix = generate_activation_key()
                path_to_link = '/media/' + prefix
                dest = BASE_DIR + path_to_link
                os.symlink(stock_to_show.pic.path, dest)

                # create symlinks wich are removed when "app" is called or user loggs out
                try:
                    symlinks = request.session["symlinks"]
                    symlinks.append(dest)
                    request.session["symlinks"] = symlinks
                except:
                    request.session["symlinks"] = [dest]

                name += ","+stock_to_show.shortfilename
                url += ","+config['SERVER']+path_to_link
                if stock_to_show.shortfilename[-3:] == '.am':
                    amira_not_supported = True
        if amira_not_supported:
            request.session['state'] = 'Rendering of Amira files is not supported. Please transfer to TIFF first.'
            next = request.GET.get('next', '')
            next = str(next)
            if next == 'storage':
                return redirect(storage)
            else:
                return redirect(app)
        else:
            name = name[1:]
            url = url[1:]
            URL=config['SERVER']+"/paraview/?name=["+name+"]&url=["+url+"]"
            return HttpResponseRedirect(URL)

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
        # initialization
        initial = {}
        settings_form = SettingsForm()
        for key in settings_form.fields.keys():
            initial[key] = image.__dict__[key]
        # get data
        if request.method == 'POST':
            img = SettingsForm(request.POST)
            if img.is_valid():
                cd = img.cleaned_data
                if cd != initial:
                    for key in cd.keys():
                        image.__dict__[key] = cd[key]
                    image.validation_freq = max(1, int(cd['validation_freq']))
                    image.epochs = min(100, int(cd['epochs']))
                    image.rotate = min(180, max(0, int(cd['rotate'])))
                    image.validation_split = min(1.0, max(0.0, float(cd['validation_split'])))
                    image.stride_size = max(32, int(cd['stride_size']))
                    image.x_scale = min(512, int(cd['x_scale']))
                    image.y_scale = min(512, int(cd['y_scale']))
                    image.z_scale = min(512, int(cd['z_scale']))
                    if any([scale > 256 for scale in [image.x_scale, image.y_scale, image.z_scale]]):
                        image.stride_size = 64
                    if cd['early_stopping'] and image.validation_split == 0.0:
                        image.validation_split = 0.8
                    image.save()
                    messages.error(request, 'Your settings were changed.')
                return redirect(settings, id)
        else:
            settings_form = SettingsForm(initial=initial)
            return render(request,'settings.html',{'settings_form':settings_form,'name':image.shortfilename})

# 20. settings
@login_required
def settings_prediction(request, id):
    id = int(id)
    image = get_object_or_404(Upload, pk=id)
    if image.user == request.user and image.imageType == 4 and image.project > 0:
        initial = {}
        settings_form = SettingsPredictionForm()
        for key in settings_form.fields.keys():
            initial[key] = image.__dict__[key]
        if request.method == 'POST':
            img = SettingsPredictionForm(request.POST)
            if img.is_valid():
                cd = img.cleaned_data
                if cd != initial:
                    for key in cd.keys():
                        image.__dict__[key] = cd[key]
                    image.save()
                    messages.error(request, 'Your settings were changed.')
                return redirect(settings_prediction, id)
        else:
            settings_form = SettingsPredictionForm(initial=initial)
            return render(request,'settings.html',{'settings_form':settings_form,'name':image.shortfilename})

# 21. update_profile
@login_required
def update_profile(request):

    user = request.user
    users = None

    # allow superuser to change user
    if request.user.is_superuser:
        users = User.objects.filter()
        query = request.GET.get('search')
        if query:
            user = User.objects.get(username=query)

    # get profile
    profile = Profile.objects.get(user=user)

    # change entries or initialize form
    if request.method == 'POST':
        user_form = UserForm(request.POST, instance=user)
        if user_form.is_valid():
            cd = user_form.cleaned_data
            profile.platform = cd['platform']
            profile.notification = cd['notification']
            if request.user.is_superuser:
                profile.storage_size = cd['storage_size']
            user_form.save()
            profile.save()
            messages.error(request, 'Profile successfully updated!')
            return redirect(update_profile)
        else:
            messages.error(request, 'Please correct the error below.')
    else:
        repositories = Repository.objects.filter(users=user)
        user_form = UserForm(instance=user, initial={'notification':profile.notification, 'storage_size':profile.storage_size, 'platform':profile.platform})
        if not request.user.is_superuser:
            del user_form.fields['storage_size']

    return render(request, 'profile.html', {'user_form':user_form, 'repositories':repositories, 'user_list':users})

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

def recursive_file_permissions(path_to_dir):
    files = glob.glob(path_to_dir+'/**/*', recursive=True)
    for file in files:
        try:
            os.chmod(file, 0o770)
        except:
            pass

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
        img = StorageForm(request.POST or None, request.FILES or None)
        if request.is_ajax():
            if img.is_valid():
                newimg = Upload(pic=request.FILES['pic'])
                cd = img.cleaned_data
                newimg.imageType = 1#cd['imageType']
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
                    recursive_file_permissions(path_to_dir)
                elif extension == '.tar':
                    tar = tarfile.open(newimg.pic.path)
                    tar.extractall(path=path_to_dir)
                    tar.close()
                    recursive_file_permissions(path_to_dir)
                return redirect(storage)
    else:
        img = StorageForm()

    # update lists
    images = Upload.objects.filter(user=request.user, project=0)

    # storage size
    storage_size = request.user.profile.storage_size
    datasize = get_size(WWW_DATA_ROOT + '/images/' + request.user.username)
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
                        q = Queue('slices', connection=Redis())
                        job = q.enqueue_call(create_slices, args=(stock_to_move.pic.path, None,), timeout=-1)

                    try:
                        image_tmp = Upload.objects.get(user=request.user, project=stock_to_move.project, imageType=2)
                        path_to_slices = image_tmp.pic.path.replace("images", "sliceviewer", 1)
                        if not os.path.exists(path_to_slices):
                            q = Queue('slices', connection=Redis())
                            job = q.enqueue_call(create_slices, args=(stock_to_move.pic.path, image_tmp.pic.path,), timeout=-1)
                    except:
                        pass

                elif stock_to_move.imageType == 2:

                    try:
                        image_tmp = Upload.objects.get(user=request.user, project=stock_to_move.project, imageType=1)
                        path_to_slices = stock_to_move.pic.path.replace("images", "sliceviewer", 1)
                        if not os.path.exists(path_to_slices):
                            q = Queue('slices', connection=Redis())
                            job = q.enqueue_call(create_slices, args=(image_tmp.pic.path, stock_to_move.pic.path,), timeout=-1)
                    except:
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
                # rename django object
                stock_to_rename.shortfilename = new_name
                stock_to_rename.pic.name = 'images/' + stock_to_rename.user.username + '/' + new_name
                stock_to_rename.save()
                results = {'success':True}
    return JsonResponse(results)

# init_keras_3D
def init_keras_3D(image, label, predict, img_list=None, label_list=None,
    val_img_list=None, val_label_list=None):

    # get objects
    try:
        image = Upload.objects.get(pk=image)
        label = Upload.objects.get(pk=label)
    except Upload.DoesNotExist:
        image.status = 0
        image.save()
        message = 'Files have been removed.'
        Upload.objects.create(user=image.user, project=image.project, log=1, imageType=None, shortfilename=message)

    # check if aborted
    if image.status > 0:

        # get worker name
        worker_name = 'first_worker'
        job = get_current_job()
        if job is not None:
            worker_name = job.worker_name

        # get queue configuration
        my_env = None
        if 'first_worker' in worker_name:
            QUEUE, worker_id = 'FIRST', 1
        elif 'second_worker' in worker_name:
            QUEUE, worker_id = 'SECOND', 2
        elif 'third_worker' in worker_name:
            QUEUE, worker_id = 'THIRD', 3
            if image.queue == 1:
                image.queue = 3
                image.save()
        host = config[f'{QUEUE}_QUEUE_HOST']

        # search for failed jobs in queue
        q = Queue('check_queue', connection=Redis())
        job = q.enqueue_call(init_check_queue, args=(image.id, worker_id,), timeout=-1)

        # set status to processing
        image.status = 2
        if predict:
            image.message = 'Processing'
        elif label.automatic_cropping:
            image.message = 'Train automatic cropping'
        else:
            image.message = 'Progress 0%'
        image.save()

        # number of gpus or list of gpu ids
        if type(config[f'{QUEUE}_QUEUE_NGPUS'])==list:
            list_of_ids = config[f'{QUEUE}_QUEUE_NGPUS']
            gpu_ids = ''
            for id in list_of_ids:
                gpu_ids = gpu_ids + f'{id},'
            gpu_ids = gpu_ids[:-1]
            my_env = os.environ.copy()
            my_env['CUDA_VISIBLE_DEVICES'] = gpu_ids
        elif type(config[f'{QUEUE}_QUEUE_NGPUS'])==int:
            ngpus = config[f'{QUEUE}_QUEUE_NGPUS']
            gpu_ids = ''
            for id in range(ngpus):
                gpu_ids = gpu_ids + f'{id},'
            gpu_ids = gpu_ids[:-1]
            my_env = os.environ.copy()
            my_env['CUDA_VISIBLE_DEVICES'] = gpu_ids

        # change working directory
        cwd = BASE_DIR + '/biomedisa_features/'

        # command
        if predict:
            cmd = ['python3','biomedisa_deeplearning.py',str(image.id),str(label.id),'-p','-de','-sc']
        else:
            cmd = ['python3','biomedisa_deeplearning.py',str(image.id),str(label.id),'-t','-de',
                   '-il',str(img_list),'-ll',str(label_list),'-sc','-tt']
            if val_img_list and val_label_list:
                cmd = cmd + ['-vi',val_img_list,'-vl',val_label_list]

        if host:
            cmd[1] = cwd+'biomedisa_deeplearning.py'
            cmd = ['ssh', host] + cmd
            p = subprocess.Popen(cmd)
        else:
            p = subprocess.Popen(cmd, cwd=cwd, env=my_env)

        # wait for process to finish
        p.wait()

        # stop processing
        image.path_to_model = ''
        image.status = 0
        image.pid = 0
        image.save()

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
        model = 0
        for img in images:
            if img.imageType == 4:
                model = img.id

        # predict segmentation
        if model > 0:
            for img in images:
                if img.status > 0:
                    request.session['state'] = 'Image is already being processed.'
                elif img.imageType == 1:

                    # two processing queues
                    if config['SECOND_QUEUE']:
                        q1 = Queue('first_queue', connection=Redis())
                        q2 = Queue('second_queue', connection=Redis())
                        w1 = Worker.all(queue=q1)[0]
                        w2 = Worker.all(queue=q2)[0]
                        lenq1 = len(q1)
                        lenq2 = len(q2)

                        if lenq1 > lenq2 or (lenq1==lenq2 and w1.state=='busy' and w2.state=='idle'):
                            queue_short = 'B'
                            job = q2.enqueue_call(init_keras_3D, args=(img.id, model, True), timeout=-1)
                            lenq = len(q2)
                            img.queue = 2
                        else:
                            queue_short = 'A'
                            job = q1.enqueue_call(init_keras_3D, args=(img.id, model, True), timeout=-1)
                            lenq = len(q1)
                            img.queue = 1

                    # single processing queue
                    else:
                        queue_short = 'A'
                        q = Queue('first_queue', connection=Redis())
                        job = q.enqueue_call(init_keras_3D, args=(img.id, model, True), timeout=-1)
                        lenq = len(q)
                        img.queue = 1

                    if lenq == 0:
                        img.message = 'Processing'
                    else:
                        img.message = f'Queue {queue_short} position {lenq} of {lenq}'
                    img.status = 1
                    img.job_id = job.id
                    img.save()

                elif img.imageType in [2,3]:
                    request.session['state'] = 'No vaild image selected.'
        else:
            request.session['state'] = 'No vaild network selected.'

    # train a neural network
    elif int(action) == 4:

        todo = request.GET.getlist('selected')
        images = Upload.objects.filter(pk__in=todo)

        # get list of images and labels
        img_list, label_list = '', ''
        val_img_list, val_label_list = '', ''
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
                if label.validation_data:
                    val_img_list += raw.pic.path + ';'
                    val_label_list += label.pic.path + ';'
                else:
                    raw_out = raw
                    label_out = label
                    img_list += raw.pic.path + ';'
                    label_list += label.pic.path + ';'

        # train neural network
        if not img_list:
            request.session['state'] = 'No usable image and label combination selected.'
        elif raw_out.status > 0:
            request.session['state'] = 'Image is already being processed.'
        else:
            if config['THIRD_QUEUE']:
                queue_name, queue_short = 'third_queue', 'C'
                raw_out.queue = 3
            else:
                queue_name, queue_short = 'first_queue', 'A'
                raw_out.queue = 1
            q = Queue(queue_name, connection=Redis())
            job = q.enqueue_call(init_keras_3D, args=(raw_out.id, label_out.id, False,
                                 img_list, label_list, val_img_list, val_label_list), timeout=-1)
            lenq = len(q)
            raw_out.job_id = job.id
            if lenq == 0:
                if label_out.automatic_cropping:
                    raw_out.message = 'Train automatic cropping'
                else:
                    raw_out.message = 'Progress 0%'
            else:
                raw_out.message = f'Queue {queue_short} position {lenq} of {lenq}'
            raw_out.status = 1
            raw_out.save()

    # duplicate file
    elif int(action) == 6:

        todo = request.GET.getlist('selected')
        for id in todo:
            id = int(id)
            stock_to_duplicate = get_object_or_404(Upload, pk=id)
            if stock_to_duplicate.user == request.user:

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

                    # create unique filename
                    path_to_data = unique_file_path(img.pic.path, img.user.username, WWW_DATA_ROOT+'/')
                    pic_path = 'images/' + img.user.username + '/' + os.path.basename(path_to_data)
                    new_short_name = os.path.basename(path_to_data)

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

                    # copy slices
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
    elif int(action) in [7,8,11]:
        todo = request.GET.getlist('selected')
        for id in todo:
            id = int(id)
            img = get_object_or_404(Upload, pk=id)
            if img.user == request.user:
                if img.status > 0:
                    request.session['state'] = img.shortfilename + ' is already being processed.'
                else:
                    q = Queue('process_image', connection=Redis())
                    if int(action) == 7:
                        job = q.enqueue_call(convert_image, args=(img.id,), timeout=-1)
                    elif int(action) == 8:
                        job = q.enqueue_call(smooth_image, args=(img.id,), timeout=-1)
                    elif int(action) == 11:
                        job = q.enqueue_call(convert_to_stl, args=(img.id,), timeout=-1)
                    lenq = len(q)
                    img.job_id = job.id
                    if lenq == 0:
                        img.status = 2
                        img.message = 'Processing'
                    else:
                        img.status = 1
                        img.message = f'Queue E position {lenq} of {lenq}'
                    img.queue = 5
                    img.save()

    # switch image type
    elif int(action) == 9:
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
            with open(BASE_DIR + '/log/logfile.txt', 'a') as logfile:
                print('%s reset %s %s' %(time.ctime(), img.user, img.shortfilename), file=logfile)
                img.status = 0
                img.pid = 0
                img.save()
    return redirect(app)

# 27. app
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
            for directory in ['images', 'sliceviewer']:
                path_to_data = WWW_DATA_ROOT + '/' + directory + '/' + str(profile.user.username)

                if not os.path.isdir(path_to_data):
                    os.makedirs(path_to_data)
                    try:
                        os.chmod(path_to_data, 0o770)
                    except:
                        pass

    # run biomedisa features
    action = request.GET.get('action')
    if action:
        if int(action)==10:
            selected = request.GET.getlist('selected')
            if selected:
                s = selected[0]
                for i in selected[1:]:
                    s += ","+i
                return redirect(reverse(visualization)+"?selected="+s)
        else:
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
                recursive_file_permissions(path_to_dir)
            elif extension == '.tar':
                tar = tarfile.open(newimg.pic.path)
                tar.extractall(path=path_to_dir)
                tar.close()
                recursive_file_permissions(path_to_dir)

            # create slices
            if newimg.imageType == 1:

                path_to_slices = newimg.pic.path.replace("images", "sliceviewer", 1)
                if not os.path.exists(path_to_slices):
                    q = Queue('slices', connection=Redis())
                    job = q.enqueue_call(create_slices, args=(newimg.pic.path, None,), timeout=-1)

                try:
                    tmp = Upload.objects.get(user=request.user, project=newimg.project, imageType=2)
                    path_to_slices = tmp.pic.path.replace("images", "sliceviewer", 1)
                    if not os.path.exists(path_to_slices):
                        q = Queue('slices', connection=Redis())
                        job = q.enqueue_call(create_slices, args=(newimg.pic.path, tmp.pic.path,), timeout=-1)
                except:
                    pass

            elif newimg.imageType == 2:

                try:
                    tmp = Upload.objects.get(user=request.user, project=newimg.project, imageType=1)
                    path_to_slices = newimg.pic.path.replace("images", "sliceviewer", 1)
                    if not os.path.exists(path_to_slices):
                        q = Queue('slices', connection=Redis())
                        job = q.enqueue_call(create_slices, args=(tmp.pic.path, newimg.pic.path,), timeout=-1)
                except:
                    pass

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
            process_list += ";" + str(image.id) + ":" + str(image.status) + ":" + str(image.message)

        # one and only one final object is allowed to be active
        if image.final:
            tmp = [x for x in images if image.friend==x.friend]
            if not any(x.active for x in tmp):
                tmp[0].active = 1
                tmp[0].save()

        # update queue position
        if image.status == 1:

            if image.queue == 1:
                queue_name, queue_short = 'first_queue', 'A'
            elif image.queue == 2:
                queue_name, queue_short = 'second_queue', 'B'
            elif image.queue == 3:
                queue_name, queue_short = 'third_queue', 'C'
            elif image.queue == 4:
                queue_name, queue_short = 'acwe', 'D'
            elif image.queue == 5:
                queue_name, queue_short = 'process_image', 'E'

            id_to_check = image.job_id
            new_message = image.message
            q = Queue(queue_name, connection=Redis())
            if id_to_check in q.job_ids:
                i = q.job_ids.index(id_to_check)
                new_message = f'Queue {queue_short} position {i+1} of {len(q)}'

            if image.message != new_message:
                image.message = new_message
                image.save()

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
    datasize = get_size(WWW_DATA_ROOT + '/images/' + request.user.username)
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

# return error
def return_error(img, error_message):
    '''
    Resets the image object when an error occurs. While _error_ in biomedisa_helper.py does not reset the object.

    Parameter
    ---------
    img: object
        object of the image file
    error_message: string
        name of the error
    '''
    img.status = 0
    img.pid = 0
    img.save()
    Upload.objects.create(user=img.user, project=img.project, log=1, imageType=None, shortfilename=error_message)
    path_to_logfile = BASE_DIR + '/log/logfile.txt'
    with open(path_to_logfile, 'a') as logfile:
        print('%s %s %s %s' %(time.ctime(), img.user.username, img.shortfilename, error_message), file=logfile)
    send_error_message(img.user.username, img.shortfilename, error_message)

# search for failed jobs
def init_check_queue(id, processing_queue):
    images = Upload.objects.filter(status=2, queue=processing_queue)
    for img in images:
        if img.id != id:
            return_error(img, 'Something went wrong. Please restart.')

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
        list_of_users = str(list_of_users)

        # share demo file
        demo = request.GET.get('demo')

        # get object
        if demo:
            id = str(id)
            demo_id = User.objects.get(username='demo')
            stock_to_share = Upload.objects.filter(user_id=demo_id, shortfilename=id)[0]
        else:
            id = int(id)
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

                    if stock_to_share.final:
                        if demo:
                            images = Upload.objects.filter(user=demo_id, friend=stock_to_share.friend)
                        else:
                            images = Upload.objects.filter(user=request.user, friend=stock_to_share.friend)
                    else:
                        images = [stock_to_share]

                    for k, img in enumerate(images):

                        # new file path
                        pic_path = 'images/' + new_user_name + '/' + img.shortfilename

                        # rename image if path already exists
                        if os.path.exists(WWW_DATA_ROOT+'/'+pic_path):
                            path_to_data = unique_file_path(pic_path, new_user_name, WWW_DATA_ROOT+'/')
                            pic_path = 'images/' + new_user_name + '/' + os.path.basename(path_to_data)

                        # create object
                        if img.final:
                            if k == 0:
                                ref_img = Upload.objects.create(pic=pic_path, user=user_id, project=0, imageType=img.imageType,
                                            shortfilename=os.path.basename(pic_path), final=img.final, shared=1, shared_by=shared_by,
                                            shared_path=img.pic.path, active=img.active)
                                ref_img.friend = ref_img.id
                                ref_img.save()
                            else:
                                Upload.objects.create(pic=pic_path, user=user_id, project=0, imageType=img.imageType,
                                    shortfilename=os.path.basename(pic_path), final=img.final, shared=1, shared_by=shared_by,
                                    shared_path=img.pic.path, friend=ref_img.id, active=img.active)
                        else:
                            Upload.objects.create(pic=pic_path, user=user_id, project=0, imageType=img.imageType,
                                shortfilename=os.path.basename(pic_path), final=img.final, shared=1,
                                shared_by=shared_by, shared_path=img.pic.path)

                    if shared_id != user_id:
                        q = Queue('share_notification', connection=Redis())
                        job = q.enqueue_call(send_share_notify, args=(user_id.username, os.path.basename(pic_path), shared_by,), timeout=-1)
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

                    # rename image if path already exists
                    if os.path.exists(img.pic.path):
                        path_to_data = unique_file_path(img.pic.path, img.user.username, WWW_DATA_ROOT+'/')
                        pic_path = 'images/' + img.user.username + '/' + os.path.basename(path_to_data)
                        img.shortfilename = os.path.basename(path_to_data)
                        img.pic.name = pic_path
                        img.save()

                    # copy file
                    os.link(img.shared_path, img.pic.path)

                    # copy untared or unzipped data
                    if extension in ['.zip', '.tar', '.tar.gz']:
                        path_to_src = img.shared_path[:-len(extension)]
                        path_to_dest = img.pic.path[:-len(extension)]
                        if os.path.exists(path_to_src):
                            copytree(path_to_src, path_to_dest, copy_function=os.link)

                    # copy slices
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
def download_demo(request):
    if request.method == 'GET':
        id = request.GET.get('id')
        demo_files = glob.glob(BASE_DIR + '/media/data/*')
        demo_files = [os.path.basename(x) for x in demo_files]
        max_str = max(demo_files, key=len)
        if id[:len(max_str)] in demo_files:
            path_to_file = BASE_DIR + '/media/data/' + id
            wrapper = FileWrapper(open(path_to_file, 'rb'))
            imgsize = os.path.getsize(path_to_file)
            if imgsize < 5000000000:
                response = HttpResponse(wrapper, content_type='%s' %(id))
            else:
                response = StreamingHttpResponse(wrapper, content_type='%s' %(id))
            response['Content-Disposition'] = 'attachment; filename="%s"' %(id)
            response['Content-Length'] = imgsize
            return response

# 39. visualization_demo
def visualization_demo(request):
    if request.method == 'GET':
        id = request.GET.get('id')
        demo_files = glob.glob(BASE_DIR + '/media/paraview/*')
        demo_files = [os.path.basename(x) for x in demo_files]
        max_str = max(demo_files, key=len)
        if id[:len(max_str)] in demo_files:
            url = config['SERVER'] + '/media/paraview/' + id
            URL = config['SERVER'] + "/paraview/?name=["+id+"]&url=["+url+"]"
            return HttpResponseRedirect(URL)

# 40. sliceviewer_demo
def sliceviewer_demo(request):
    if request.method == 'GET':
        id = request.GET.get('id')
        demo_files = glob.glob(BASE_DIR + '/media/data/*')
        demo_files = [os.path.basename(x) for x in demo_files]
        max_str = max(demo_files, key=len)
        if id[:len(max_str)] in demo_files:
            path_to_slices = "/media/sliceviewer/" + id
            full_path = BASE_DIR + path_to_slices
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
        obj = str(request.GET.get('object'))[:11]
        try:
            # delete tomographic data object
            if obj == 'tomographic':
                stock_to_delete = get_object_or_404(TomographicData, pk=id)
                if request.user in stock_to_delete.specimen.repository.users.all():
                    stock_to_delete.delete()
            # delete specimen object
            elif obj == 'specimen':
                stock_to_delete = get_object_or_404(Specimen, pk=id)
                if request.user in stock_to_delete.repository.users.all():
                    stock_to_delete.delete()
            # delete upload object
            else:
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
def init_random_walk(image, label):

    # get objects
    try:
        image = Upload.objects.get(pk=image)
        label = Upload.objects.get(pk=label)
    except Upload.DoesNotExist:
        image.status = 0
        image.save()
        message = 'Files have been removed.'
        Upload.objects.create(user=image.user, project=image.project, log=1, imageType=None, shortfilename=message)

    # check if aborted
    if image.status > 0:

        # create biomedisa
        bm = Biomedisa()
        bm.success = True

        # get worker name
        worker_name = 'first_worker'
        job = get_current_job()
        if job is not None:
            worker_name = job.worker_name

        # get queue configuration
        my_env = None
        if 'first_worker' in worker_name:
            QUEUE, queue_id = 'FIRST', 1
        elif 'second_worker' in worker_name:
            QUEUE, queue_id = 'SECOND', 2
        elif 'third_worker' in worker_name:
            QUEUE, queue_id = 'THIRD', 3
            if image.queue == 1:
                image.queue = 3
                image.save()
        host = config[f'{QUEUE}_QUEUE_HOST']
        cluster = False
        if f'{QUEUE}_QUEUE_CLUSTER' in config:
            cluster = config[f'{QUEUE}_QUEUE_CLUSTER']

        # search for failed jobs in queue
        q = Queue('check_queue', connection=Redis())
        job = q.enqueue_call(init_check_queue, args=(image.id, queue_id,), timeout=-1)

        # set status to processing
        image.status = 2
        image.message = 'Processing'
        image.save()

        # get platform
        if image.user.profile.platform:
            bm.platform = image.user.profile.platform
        else:
            bm.platform = None

        # check if platform is available
        if not host:
            bm = _get_platform(bm)

            # stop process
            if bm.success == False:
                return_error(image, f'No {bm.platform} device found.')
                raise Exception(f'No {bm.platform} device found.')

        # number of gpus or list of gpu ids
        if type(config[f'{QUEUE}_QUEUE_NGPUS'])==list:
            list_of_ids = config[f'{QUEUE}_QUEUE_NGPUS']
            ngpus = str(len(list_of_ids))
            gpu_ids = ''
            for id in list_of_ids:
                gpu_ids = gpu_ids + f'{id},'
            gpu_ids = gpu_ids[:-1]
            my_env = os.environ.copy()
            my_env['CUDA_VISIBLE_DEVICES'] = gpu_ids
        else:
            ngpus = str(config[f'{QUEUE}_QUEUE_NGPUS'])
            if ngpus == 'all':
                if host:
                    return_error(image, 'Number of GPUs must be given if running on a remote host.')
                    raise Exception('Number of GPUs must be given if running on a remote host.')
                else:
                    ngpus = str(bm.available_devices)

        # command
        cmd = ['mpiexec', '-np', ngpus, 'python3', 'biomedisa_interpolation.py', str(image.id), str(label.id), '-de']

        # specifiy platform
        if bm.platform:
            cmd.append('-p')
            cmd.append(bm.platform)
            if bm.platform.split('_')[-1] == 'CPU':
                cmd = cmd[3:]

        # change working directory
        cwd = BASE_DIR + '/biomedisa_features/'
        workers_host = BASE_DIR + '/log/workers_host'

        # run
        if cluster and 'mpiexec' in cmd:
            cmd.insert(3, '--hostfile')
            cmd.insert(4, workers_host)
        if host:
            cmd.insert(0, 'ssh')
            cmd.insert(1, host)
            cmd[cmd.index('biomedisa_interpolation.py')] = cwd+'biomedisa_interpolation.py'
            p = subprocess.Popen(cmd)
        else:
            p = subprocess.Popen(cmd, cwd=cwd, env=my_env)

        # wait for process to finish
        p.wait()

        # stop processing
        image.status = 0
        image.pid = 0
        image.save()

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

            if raw.user==request.user and label.user==request.user and raw.status==0:

                # two processing queues
                if config['SECOND_QUEUE']:
                    q1 = Queue('first_queue', connection=Redis())
                    q2 = Queue('second_queue', connection=Redis())
                    w1 = Worker.all(queue=q1)[0]
                    w2 = Worker.all(queue=q2)[0]
                    lenq1 = len(q1)
                    lenq2 = len(q2)

                    if lenq1 > lenq2 or (lenq1==lenq2 and w1.state=='busy' and w2.state=='idle'):
                        queue_name = 'B'
                        job = q2.enqueue_call(init_random_walk, args=(raw.id, label.id), timeout=-1)
                        lenq = len(q2)
                        raw.queue = 2
                    else:
                        queue_name = 'A'
                        job = q1.enqueue_call(init_random_walk, args=(raw.id, label.id), timeout=-1)
                        lenq = len(q1)
                        raw.queue = 1

                # single processing queue
                else:
                    queue_name = 'A'
                    q1 = Queue('first_queue', connection=Redis())
                    job = q1.enqueue_call(init_random_walk, args=(raw.id, label.id), timeout=-1)
                    lenq = len(q1)
                    raw.queue = 1

                raw.job_id = job.id
                if lenq == 0:
                    raw.message = 'Processing'
                else:
                    raw.message = f'Queue {queue_name} position {lenq} of {lenq}'
                raw.status = 1
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
        if host:
            subprocess.Popen(['ssh', host, 'kill', str(id)])
        else:
            subprocess.Popen(['kill', str(id)])
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

                # stop keras softly
                if 'Progress' in image_to_stop.message:
                    percent = image_to_stop.message.split(' ')[1]
                    percent = percent.strip('%,')
                    percent = float(percent)
                    if percent > 0:
                        image_to_stop.status = 3
                        image_to_stop.message = 'Process stopped. Please wait.'
                        image_to_stop.save()
                        results = {'success':True}
                        return JsonResponse(results)

                # remove from queue
                if image_to_stop.queue == 1:
                    queue_name = 'first_queue'
                elif image_to_stop.queue == 2:
                    queue_name = 'second_queue'
                elif image_to_stop.queue == 3:
                    queue_name = 'third_queue'
                elif image_to_stop.queue == 4:
                    queue_name = 'acwe'
                elif image_to_stop.queue == 5:
                    queue_name = 'process_image'
                q = Queue(queue_name, connection=Redis())
                id_to_check = image_to_stop.job_id
                if id_to_check in q.job_ids:
                    job = q.fetch_job(id_to_check)
                    job.delete()

                # kill running process
                if image_to_stop.status in [2,3] and image_to_stop.pid > 0:
                    q = Queue('stop_job', connection=Redis())
                    job = q.enqueue_call(stop_running_job, args=(image_to_stop.pid, image_to_stop.queue), timeout=-1)
                    image_to_stop.pid = 0

                # remove trained network
                if image_to_stop.path_to_model:
                    if os.path.isfile(image_to_stop.path_to_model):
                        os.remove(image_to_stop.path_to_model)
                    image_to_stop.path_to_model = ''

                # reset image
                image_to_stop.status = 0
                image_to_stop.save()

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
        for directory in ['images', 'sliceviewer']:
            path_to_data = WWW_DATA_ROOT + '/' + directory + '/' + request.user.username
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
def send_notification(username, image_name, time_str, server_name, train=False, predict=False):

    if train:
        info = 'trained a neural network for'
        recipients = [config['EMAIL']]
    else:
        info = 'finished the segmentation of'
        recipients = []

    user = User.objects.get(username=username)
    if user.first_name:
        username = user.first_name

    if user.profile.notification and user.email and user.email not in recipients:
        recipients.append(user.email)

    for recipient in recipients:

        # prepare data package
        datas={}
        datas['email'] = recipient
        if train or predict:
            datas['email_path'] = '/DeepLearningCompleted.txt'
        else:
            datas['email_path'] = '/InterpolationCompleted.txt'
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
            status[int(x[0])] = (int(x[1]), x[2]) # (image status, image message)

        reload = False
        images = Upload.objects.filter(pk__in=status.keys())
        for im in images:
            if im.status != status[im.id][0] or im.message != status[im.id][1]:
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

