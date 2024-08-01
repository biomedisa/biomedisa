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

from __future__ import unicode_literals

import django
django.setup()
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, StreamingHttpResponse, JsonResponse, HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from django.template import Context
from django.db.models import Q

from biomedisa_app.config import config
from biomedisa.features.biomedisa_helper import unique_file_path
from biomedisa.settings import BASE_DIR, PRIVATE_STORAGE_ROOT
from biomedisa_app.models import (Upload,
    Repository, Specimen, SpecimenForm, TomographicData, TomographicDataForm,
    ProcessedData, RepositoryUser)

from wsgiref.util import FileWrapper
import numpy as np
from PIL import Image
import os, sys
from shutil import copytree
import shutil
import glob

def repository(request, id=1):
    '''
    Shows repository.

    Parameters
    ----------
    id: int
        Id of the repository
    '''
    try:
        state = request.session['state']
    except KeyError:
        state = None
    # get repository
    repository = get_object_or_404(Repository, pk=int(id))
    # search for specimen containing query and sort alphabetically
    query = request.GET.get('search')
    show_all = True
    if query:
        specimens = Specimen.objects.filter(Q(internal_id__icontains=query) | Q(subfamily__icontains=query) | Q(genus__icontains=query)\
         | Q(species__icontains=query) | Q(caste__icontains=query), repository=repository).order_by('name_recommended').values()
    else:
        specimens = Specimen.objects.filter(repository=repository).order_by('name_recommended').values()
        show_all = request.GET.get('show_all') == 'True'
        if not show_all:
            specimens = specimens[:100]
    all_specimens = Specimen.objects.filter(repository=repository)
    # featured img
    imshape_x, imshape_y = None, None
    if repository.featured_img:
        imshape = np.asarray(Image.open(BASE_DIR + f'/biomedisa_app/static/{repository.featured_img}')).shape
        imshape_x = 960
        imshape_y = int(imshape[0]/imshape[1]*960)
    return render(request, 'repository.html', {'state':state, 'specimens':specimens, 'repository':repository,
                'all_specimens':all_specimens, 'featured_img':repository.featured_img,
                'featured_img_width':imshape_x, 'featured_img_height':imshape_y,
                'show_all': show_all})

@login_required
def share_repository(request):
    '''
    Add user to repository users.
    Allows to edit the repository.
    '''
    results = {'success':False}
    if request.method == 'GET':
        id = int(request.GET.get('id'))
        username = str(request.GET.get('username'))
        repository = get_object_or_404(Repository, pk=id)
        if RepositoryUser.objects.get(user=request.user, repository=repository).can_share:
            if User.objects.filter(username=username).exists():
                user_to_add = User.objects.get(username=username)
                if not RepositoryUser.objects.filter(user=user_to_add, repository=repository).exists():
                    RepositoryUser.objects.create(user=user_to_add, repository=repository, can_edit=True)
                    results = {'success':True, 'msg':'Repository shared successfully.'}
                    q = Queue('share_notification', connection=Redis())
                    job = q.enqueue_call(repository_share_notify, args=(username, repository.name, request.user.username,), timeout=-1)
                else:
                    results = {'success':True, 'msg':f'{username} is already member of {repository.name}'}
            else:
                results = {'success':True, 'msg':'User does not exist.'}
        else:
            results = {'success':True, 'msg':'No permission to share.'}
    return JsonResponse(results)

def repository_share_notify(username, repository, shared_by):
    '''
    Notifies user that the repository was shared with them.
    '''
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
    '''
    Unsubscribe from repository.

    Parameters
    ----------
    id: int
        Id of the repository
    '''
    results = {'success':False}
    repository = get_object_or_404(Repository, pk=id)
    user_to_remove = User.objects.get(username=request.user)
    if request.user in repository.users.all():
        repository.users.remove(user_to_remove)
        repository.save()
        results = {'success':True}
    return JsonResponse(results)

def specimen_info(request, id):
    '''
    Show and edit metadata of an individual specimen.

    Parameters
    ----------
    id: int
        Id of the repository
    '''
    id = int(id)
    specimen = get_object_or_404(Specimen, pk=id)
    # initialization
    initial = {}
    specimen_form = SpecimenForm()
    for key in specimen_form.fields.keys():
        initial[key] = specimen.__dict__[key]
    # user info
    user_can_edit = False
    if request.user.is_authenticated \
        and RepositoryUser.objects.filter(user=request.user, repository=specimen.repository).exists() \
        and RepositoryUser.objects.get(user=request.user, repository=specimen.repository).can_edit:
        user_can_edit = True
    # get data
    if request.method=='POST' and user_can_edit:
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
        images = None
        imshape = (0,0)
        if ProcessedData.objects.filter(specimen=specimen, imageType=1).exists():
            processed_data = ProcessedData.objects.filter(specimen=specimen, imageType=1)[0]
            path_to_slices = BASE_DIR + os.path.splitext('/media/' + processed_data.pic.name)[0] + '_lowres'
            if os.path.exists(path_to_slices):
                images = sorted(glob.glob(path_to_slices + '/*.png'))
                imshape = np.asarray(Image.open(images[0])).shape
                imshape_x = 400
                imshape_y = int(imshape[0]/imshape[1]*400)
                imshape = (imshape_y,imshape_x)
                images = [img.replace(BASE_DIR,'') for img in images]
        # specimen form
        specimen_form = SpecimenForm(initial=initial)
        name = specimen.internal_id if not any([specimen.name_recommended, specimen.subfamily, specimen.caste, specimen.specimen_code]) else "{name_recommended} | {subfamily} | {caste} | {specimen_code}".format(name_recommended=specimen.name_recommended, subfamily=specimen.subfamily, caste=specimen.caste, specimen_code=specimen.specimen_code)
        #tomographic_data = TomographicData.objects.filter(specimen=specimen)
        processed_data = ProcessedData.objects.filter(specimen=specimen)
        sketchfab_id = specimen.sketchfab
        return render(request, 'specimen_info.html', {'specimen_form':specimen_form,#'tomographic_data':tomographic_data,
                                                      'processed_data':processed_data,'name':name,'specimen':specimen,
                                                      'imshape_x':imshape[1], 'imshape_y':imshape[0],'paths':images,
                                                      'user_can_edit':user_can_edit})

def tomographic_info(request, id):
    '''
    Show and edit tomo and processed data related to an individual specimen.

    Parameters
    ----------
    id: int
        Id of the repository
    '''
    id = int(id)
    tomographic_data = get_object_or_404(TomographicData, pk=id)
    # initialization
    initial = {}
    tomographic_form = TomographicDataForm()
    for key in tomographic_form.fields.keys():
        initial[key] = tomographic_data.__dict__[key]
    # user info
    user_can_edit = False
    if request.user.is_authenticated \
        and RepositoryUser.objects.filter(user=request.user, repository=tomographic_data.specimen.repository).exists() \
        and RepositoryUser.objects.get(user=request.user, repository=tomographic_data.specimen.repository).can_edit:
        user_can_edit = True
    if request.method=='POST' and user_can_edit:
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
        images = None
        imshape = (0,0)
        # preview tomographic data
        path_to_slices = BASE_DIR + os.path.splitext('/media/' + tomographic_data.pic.name)[0]
        if os.path.exists(path_to_slices):
            images = sorted(glob.glob(path_to_slices + '/*.png'))
            imshape = np.asarray(Image.open(images[0])).shape
            imshape_x = 400
            imshape_y = int(imshape[0]/imshape[1]*400)
            imshape = (imshape_y,imshape_x)
            images = [img.replace(BASE_DIR,'') for img in images]
        # tomographic form
        tomographic_form = TomographicDataForm(initial=initial)
        return render(request, 'tomographic_info.html', {'tomographic_form':tomographic_form,
                                                         'tomographic_data':tomographic_data,
                                                         'user_can_edit':user_can_edit,
                                                         'paths':images, 'imshape_x':imshape[1],
                                                         'imshape_y':imshape[0]})

def sliceviewer_repository(request):
    if request.method == 'GET':
        id = int(request.GET.get('id'))
        obj = str(request.GET.get('object'))[:11]
        #if obj == 'tomographic':
        #    specimen = get_object_or_404(ProcessedData, pk=id)
        #    tomographic_data = ProcessedData.objects.filter(specimen=specimen, )
        if obj == 'processed':
            tomographic_data = get_object_or_404(ProcessedData, pk=id)
        elif obj == 'specimen':
            specimen = get_object_or_404(Specimen, pk=id)
            tomographic_data = ProcessedData.objects.filter(specimen=specimen, imageType=1)[0]
        path_to_slices = BASE_DIR + '/media/' + tomographic_data.pic.name.replace('.tif','')
        images = sorted(glob.glob(path_to_slices + '/*.png'))
        imshape = np.asarray(Image.open(images[0])).shape
        images = [img.replace(BASE_DIR,'') for img in images]
        return render(request, 'sliceviewer.html', {'paths':images, 'imshape_x':imshape[1], 'imshape_y':imshape[0]})

def visualization_repository(request):
    if request.method == 'GET':
        id = int(request.GET.get('id'))
        obj = str(request.GET.get('object'))[:11]
        if obj == 'processed':
            specimen = get_object_or_404(ProcessedData, pk=id).specimen
        elif obj == 'specimen':
            specimen = get_object_or_404(Specimen, pk=id)
        path_to_link = f'/media/antscan/processed/{specimen.magnification}/{specimen.internal_id}.stl'
        name = specimen.internal_id + '.stl'
        url = config['SERVER'] + path_to_link
        URL = config['SERVER'] + "/paraview/?name=["+name+"]&url=["+url+"]"
        return HttpResponseRedirect(URL)

def download_repository(request):
    if request.method == 'GET':
        id = int(request.GET.get('id'))
        obj = str(request.GET.get('object'))[:11]
        if obj == 'processed':
            tomographic_data = get_object_or_404(ProcessedData, pk=id)
        elif obj == 'specimen':
            specimen = get_object_or_404(Specimen, pk=id)
            tomographic_data = ProcessedData.objects.filter(specimen=specimen, imageType=1)[0]
        filename = tomographic_data.pic.name
        path_to_file = tomographic_data.pic.path
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
    '''
    Share repo data with user account.

    Parameters
    ----------
    id: int
        Id of the repository
    '''
    results = {'success':False}
    if request.method == 'GET':
        id = int(request.GET.get('id'))
        processed_data = get_object_or_404(ProcessedData, pk=id)

        # new file path
        shortfilename = processed_data.pic.name.replace('/','_')
        pic_path = 'images/' + request.user.username + '/' + shortfilename

        # rename image if path already exists
        if os.path.exists(PRIVATE_STORAGE_ROOT + '/' + pic_path):
            path_to_data = unique_file_path(pic_path)
            pic_path = 'images/' + request.user.username + '/' + os.path.basename(path_to_data)
            shortfilename = os.path.basename(path_to_data)

        # create object
        img = Upload.objects.create(pic=pic_path, user=request.user, project=0, shortfilename=shortfilename)

        # copy file
        shutil.copy2(processed_data.pic.path, img.pic.path)

        # copy slices
        path_to_src = os.path.splitext(processed_data.pic.path)[0]
        if os.path.exists(path_to_src):
            path_to_dest = img.pic.path.replace('images', 'sliceviewer', 1)
            copytree(path_to_src, path_to_dest)

        results = {'success':True, 'msg':'Successfully shared data.'}
    return JsonResponse(results)

