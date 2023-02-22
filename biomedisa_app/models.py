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

from django.db import models
from django import forms
from django.contrib.auth.models import User
from biomedisa_app.config import config
from django.dispatch import receiver
from django.core.exceptions import ValidationError
from django.core.validators import RegexValidator
from django.contrib.auth.password_validation import validate_password
from django.core.files.storage import FileSystemStorage
from django.utils.deconstruct import deconstructible
import shutil
import os

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    storage_size = models.IntegerField(default=100)
    notification = models.BooleanField(default=True)
    activation_key = models.TextField(null=True)
    key_expires = models.DateTimeField(null=True)
    platform = models.CharField(default='', max_length=20)

class UserForm(forms.ModelForm):
    notification = forms.BooleanField(required=False)
    storage_size = forms.IntegerField(required=False)
    platform = forms.CharField(required=False)
    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email', 'platform', 'storage_size', 'notification')

def user_directory_path(instance, filename):
    filename = filename.encode('ascii', 'ignore').decode()
    filename = os.path.basename(filename)
    filename, extension = os.path.splitext(filename)
    if extension == '.gz':
        filename, extension = os.path.splitext(filename)
        if extension == '.nii':
            extension = '.nii.gz'
        elif extension == '.tar':
            extension = '.tar.gz'
    filename = 'images/%s/%s' %(instance.user, filename)
    limit = 100 - len(extension)
    filename = filename[:limit] + extension
    return filename

class CustomUserCreationForm(forms.Form):
    alphanumeric = RegexValidator(r'^[0-9a-zA-Z]*$', 'Only alphanumeric characters are allowed.')
    username = forms.CharField(label='Username', min_length=4, max_length=12, validators=[alphanumeric])
    email = forms.EmailField(label='Email')
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput)
    password2 = forms.CharField(label='Confirm password', widget=forms.PasswordInput)
    institution = forms.CharField(label='Institution')
    subject = forms.CharField(label='Subject of research')
    numeric1 = forms.CharField(label='num1', widget=forms.HiddenInput())
    numeric2 = forms.CharField(label='num2', widget=forms.HiddenInput())
    verification = forms.CharField(label='Are you human?')
    message = forms.CharField(required=False, widget=forms.Textarea(attrs={'rows': 8, 'cols': 50}))

    def clean_password1(self):
        password1 = self.cleaned_data.get('password1')
        validation = validate_password(password1)
        if validation:
            raise validation
        return password1

    def clean_username(self):
        username = self.cleaned_data['username'].lower()
        r = User.objects.filter(username=username)
        if r.count():
            raise ValidationError("Username already exists.")
        return username

    def clean_email(self):
        email = self.cleaned_data['email'].lower()
        r = User.objects.filter(email=email)
        if r.count():
            raise ValidationError("Email is already in use.")
        return email

    def clean_password2(self):
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')
        if password1 and password2 and password1 != password2:
            raise ValidationError("Passwords do not match.")
        return password2

    def clean_subject(self):
        subject = self.cleaned_data.get('subject')
        institution = self.cleaned_data.get('institution')
        if institution == subject:
            raise ValidationError("Institute and subject must be different.")
        return subject

    def clean_verification(self):
        num1 = self.cleaned_data.get('numeric1')
        num2 = self.cleaned_data.get('numeric2')
        verification = self.cleaned_data.get('verification')
        try:
            if int(verification) != int(num1) + int(num2):
                raise ValidationError("Invalid input.")
            return verification
        except ValueError:
            raise ValidationError("Invalid input.")

    def save(self, datas, commit=True):
        user = User.objects.create_user(
            self.cleaned_data['username'],
            self.cleaned_data['email'],
            self.cleaned_data['password1'],
            is_active=datas['is_active'],
            is_superuser=False,
            is_staff=False
            )
        profile = Profile()
        profile.user = user
        profile.activation_key = datas['activation_key']
        profile.key_expires = datas['key_expires']
        profile.save()
        return self.cleaned_data

@deconstructible
class MyFileSystemStorage(FileSystemStorage):
    def __init__(self):
        super(MyFileSystemStorage, self).__init__(location=config['PATH_TO_BIOMEDISA'] + '/private_storage/')

class Upload(models.Model):
    pic = models.FileField("", upload_to=user_directory_path, storage=MyFileSystemStorage())
    upload_date = models.DateTimeField(auto_now_add=True, null=True)
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    CHOICES = zip( range(1,10), range(1,10) )
    project = models.IntegerField(choices=CHOICES, default=1)
    CHOICES = ( (1,'Image'), (2,'Label'), (4,'Network') )
    imageType = models.IntegerField("Type", choices=CHOICES, default=1, null=True)
    final = models.IntegerField(default=0)
    active = models.IntegerField(default=0)
    status = models.IntegerField(default=0)
    log = models.IntegerField(default=0)
    shortfilename = models.TextField(null=True)
    job_id = models.TextField(null=True)
    message = models.TextField(null=True)
    nbrw = models.IntegerField(default=10)
    sorw = models.IntegerField(default=4000)
    hashed = models.TextField(null=True)
    hashed2 = models.TextField(null=True)
    size = models.IntegerField(default=0)
    shared = models.IntegerField(default=0)
    shared_by = models.TextField(null=True)
    shared_path = models.TextField(null=True)
    friend = models.IntegerField(default=None, null=True)
    allaxis = models.BooleanField("All axes", default=False)
    diffusion = models.BooleanField(default=True)
    smooth = models.IntegerField(default=100)
    active_contours = models.BooleanField(default=True)
    ac_alpha = models.FloatField("Active contour alpha", default=1.0)
    ac_smooth = models.IntegerField("Active contour smooth", default=1)
    ac_steps = models.IntegerField("Active contour steps", default=3)
    delete_outliers = models.FloatField(default=0.9)
    fill_holes = models.FloatField(default=0.9)
    ignore = models.CharField("ignore label", default='none', max_length=20)
    predict = models.BooleanField("Predict", default=False)
    pid = models.IntegerField(default=0)
    normalize = models.BooleanField("Normalize training data (AI)", default=True)
    compression = models.BooleanField('Compress results', default=True)
    epochs = models.IntegerField("Number of epochs (AI)", default=200)
    inverse = models.BooleanField(default=False)
    only = models.CharField("compute only label", default='all', max_length=20)
    position = models.BooleanField("Consider voxel location (AI)", default=False)
    full_size = models.BooleanField("Full size preview", default=False)
    stride_size = models.IntegerField("Stride size (AI)", default=32)
    queue = models.IntegerField(default=1)
    x_scale = models.IntegerField("X Scale (AI)", default=256)
    y_scale = models.IntegerField("Y Scale (AI)", default=256)
    z_scale = models.IntegerField("Z Scale (AI)", default=256)
    balance = models.BooleanField("Balance training data (AI)", default=False)
    uncertainty = models.BooleanField(default=True)
    batch_size = models.IntegerField("Batch size (AI)", default=24)
    flip_x = models.BooleanField("Flip x-axis (AI)", default=False)
    flip_y = models.BooleanField("Flip y-axis (AI)", default=False)
    flip_z = models.BooleanField("Flip z-axis (AI)", default=False)
    rotate = models.IntegerField("Rotate (AI)", default=0)
    automatic_cropping = models.BooleanField("Automatic cropping (AI)", default=False)
    path_to_model = models.TextField(null=True)
    validation_split = models.FloatField('Validation split (AI)', default=0.0)
    early_stopping = models.BooleanField("Early stopping (AI)", default=False)
    val_tf = models.BooleanField("Validate TF accuracy (AI)", default=False)
    validation_freq = models.IntegerField("Validation frequency (AI)", default=1)
    filters = models.CharField("Network architecture (AI)", default='32-64-128-256-512-1024', max_length=30)
    resnet = models.BooleanField("ResNet convolutional blocks (AI)", default=False)

class UploadForm(forms.ModelForm):
    class Meta:
        model = Upload
        fields = ('pic', 'project', 'imageType')

class StorageForm(forms.ModelForm):
    class Meta:
        model = Upload
        fields = ('pic',)

class SettingsForm(forms.ModelForm):
    class Meta:
        model = Upload
        fields = ('allaxis', 'uncertainty', 'compression', 'normalize',
                  'automatic_cropping', 'early_stopping', 'position', 'flip_x',
                  'flip_y', 'flip_z', 'resnet', 'filters', 'rotate', 'epochs', 'batch_size',
                  'x_scale', 'y_scale', 'z_scale', 'stride_size', 'validation_split',
                  'validation_freq', 'smooth', 'delete_outliers', 'fill_holes', 'ignore', 'only')

class SettingsPredictionForm(forms.ModelForm):
    class Meta:
        model = Upload
        fields = ('compression', 'batch_size', 'stride_size', 
                  'delete_outliers', 'fill_holes')

def repository_directory_path(instance, filename):
    filename = filename.encode('ascii', 'ignore').decode()
    filename = os.path.basename(filename)
    filename, extension = os.path.splitext(filename)
    if extension == '.gz':
        filename, extension = os.path.splitext(filename)
        if extension == '.nii':
            extension = '.nii.gz'
        elif extension == '.tar':
            extension = '.tar.gz'
    limit = 100 - len(extension)
    filename = filename[:limit] + extension
    return filename

class Repository(models.Model):
    repository_alias = models.TextField(null=True)
    repository_name = models.TextField(null=True)
    repository_id = models.CharField(default=0, max_length=7)
    users = models.ManyToManyField(User, related_name="repository")
    featured_img = models.TextField(null=True)
    featured_img_width = models.TextField(null=True)
    featured_img_height = models.TextField(null=True)

class Specimen(models.Model):
    internal_id = models.CharField(null=True, max_length=255, blank=True)
    upload_date = models.DateTimeField(auto_now_add=True, null=True)
    repository = models.ForeignKey(Repository, on_delete=models.DO_NOTHING)
    subfamily = models.CharField(null=True, max_length=255, blank=True)
    genus = models.CharField(null=True, max_length=255, blank=True)
    species = models.CharField(null=True, max_length=255, blank=True)
    caste = models.CharField(null=True, max_length=255, blank=True)
    status = models.CharField(null=True, max_length=255, blank=True)
    location = models.CharField(null=True, max_length=255, blank=True)
    date = models.DateField(null=True, blank=True)
    collected_by = models.CharField(null=True, max_length=255, blank=True)
    collection_date = models.DateField(null=True, blank=True)
    determined_by = models.CharField(null=True, max_length=255, blank=True)
    collection = models.CharField(null=True, max_length=255, blank=True)
    specimen_id = models.CharField(null=True, max_length=255, blank=True)
    notes = models.TextField(null=True, blank=True)
    sketchfab = models.TextField(null=True, blank=True)
    specimen_code = models.CharField(null=True, max_length=255, blank=True)
    collection_code = models.CharField(null=True, max_length=255, blank=True)
    taxon_code = models.CharField(null=True, max_length=255, blank=True)
    lifestagesex = models.CharField(null=True, max_length=255, blank=True)
    subcaste = models.CharField(null=True, max_length=255, blank=True)
    scanning_vial_box = models.CharField(null=True, max_length=255, blank=True)
    subcaste1 = models.CharField(null=True, max_length=255, blank=True)
    tribe = models.CharField(null=True, max_length=255, blank=True)
    genus_authority = models.CharField(null=True, max_length=255, blank=True)
    source = models.CharField(null=True, max_length=255, blank=True)
    located_at = models.CharField(null=True, max_length=255, blank=True)
    owned_by = models.CharField(null=True, max_length=255, blank=True)
    insert_user = models.CharField(null=True, max_length=255, blank=True)
    method = models.CharField(null=True, max_length=255, blank=True)
    sampling_effort = models.CharField(null=True, max_length=255, blank=True)
    date_collected_start = models.CharField(null=True, max_length=255, blank=True)
    date_collected_end = models.CharField(null=True, max_length=255, blank=True)
    habitat = models.CharField(null=True, max_length=255, blank=True)
    microhabitat = models.CharField(null=True, max_length=255, blank=True)
    behavior = models.CharField(null=True, max_length=255, blank=True)
    disturbance_level = models.CharField(null=True, max_length=255, blank=True)
    country = models.CharField(null=True, max_length=255, blank=True)
    adm1 = models.CharField(null=True, max_length=255, blank=True)
    adm2 = models.CharField(null=True, max_length=255, blank=True)
    island_group = models.CharField(null=True, max_length=255, blank=True)
    island = models.CharField(null=True, max_length=255, blank=True)
    locality_name = models.CharField(null=True, max_length=255, blank=True)
    latitude = models.CharField(null=True, max_length=255, blank=True)
    longitude = models.CharField(null=True, max_length=255, blank=True)
    latlong_error = models.CharField(null=True, max_length=255, blank=True)
    elevation = models.CharField(null=True, max_length=255, blank=True)
    elevation_error = models.CharField(null=True, max_length=255, blank=True)
    biogeographic_region = models.CharField(null=True, max_length=255, blank=True)
    magnification = models.CharField(null=True, max_length=255, blank=True)
    name_recommended = models.CharField(null=True, max_length=255, blank=True)
    lts_box = models.CharField('LTS Box', null=True, max_length=255, blank=True)
    for_more_specimen = models.CharField(null=True, max_length=255, blank=True)
    specimens_left = models.CharField(null=True, max_length=255, blank=True)

class TomographicData(models.Model):
    pic = models.FileField("", upload_to=repository_directory_path)
    upload_date = models.DateTimeField(auto_now_add=True, null=True)
    specimen = models.ForeignKey(Specimen, on_delete=models.CASCADE)
    imageType = models.IntegerField("Type", default=1, null=True)
    shortfilename = models.TextField(null=True)
    facility = models.CharField(null=True, max_length=255, blank=True)
    technique = models.CharField(null=True, max_length=255, blank=True)
    projections = models.CharField(null=True, max_length=255, blank=True)
    frames_per_s = models.CharField(null=True, max_length=255, blank=True)
    filter = models.CharField(null=True, max_length=255, blank=True)
    voxel_size = models.CharField(null=True, max_length=255, blank=True)
    volume_size = models.CharField(null=True, max_length=255, blank=True)
    step_scans = models.CharField(null=True, max_length=255, blank=True)
    exposure_time_per_frame = models.CharField(null=True, max_length=255, blank=True)
    scan_tray = models.CharField(null=True, max_length=255, blank=True)

class ProcessedData(models.Model):
    pic = models.FileField("", upload_to=repository_directory_path)
    upload_date = models.DateTimeField(auto_now_add=True, null=True)
    specimen = models.ForeignKey(Specimen, on_delete=models.CASCADE)
    imageType = models.IntegerField("Type", default=1, null=True)
    shortfilename = models.TextField(null=True)

class SpecimenForm(forms.ModelForm):
    class Meta:
        model = Specimen
        widgets = {'sketchfab': forms.Textarea(attrs={'rows':1})}
        fields = ('name_recommended', 'subfamily', 'genus', 'species', 'caste', 'status',
                  'location', 'date', 'collected_by', 'collection_date',
                  'determined_by', 'collection', 'specimen_id', 'internal_id',
                  'specimen_code', 'collection_code', 'taxon_code', 'lifestagesex',
                  'subcaste', 'lts_box', 'for_more_specimen', 'specimens_left',
                  'scanning_vial_box', 'subcaste1', 'tribe',
                  'genus_authority', 'source', 'located_at', 'owned_by',
                  'insert_user', 'method', 'sampling_effort', 'date_collected_start',
                  'date_collected_end', 'habitat', 'microhabitat', 'behavior',
                  'disturbance_level', 'country', 'adm1', 'adm2',
                  'island_group', 'island', 'locality_name', 'latitude',
                  'longitude', 'latlong_error', 'elevation', 'elevation_error',
                  'biogeographic_region', 'sketchfab', 'notes')

class TomographicDataForm(forms.ModelForm):
    class Meta:
        model = TomographicData
        fields = ('facility','technique', 'projections', 'frames_per_s', 'filter',
                  'voxel_size', 'volume_size', 'step_scans', 'scan_tray', 'exposure_time_per_frame')

@receiver(models.signals.post_delete, sender=Upload)
def auto_delete_file_on_delete(sender, instance, **kwargs):
    """Deletes file from filesystem
    when corresponding `Upload` object is deleted.
    """
    if instance.pic:

        # remove preview slices
        path_to_slices = instance.pic.path.replace('images', 'sliceviewer', 1)
        if os.path.isdir(path_to_slices):
            shutil.rmtree(path_to_slices)

        # remove extracted files
        filename, extension = os.path.splitext(instance.pic.path)
        if extension == '.gz':
            filename, extension = os.path.splitext(filename)
        if extension in ['.tar','.zip'] and os.path.isdir(filename):
            shutil.rmtree(filename)

        # remove individual files
        if os.path.isfile(instance.pic.path):
            os.remove(instance.pic.path)
        elif os.path.isdir(instance.pic.path):
            shutil.rmtree(instance.pic.path)

