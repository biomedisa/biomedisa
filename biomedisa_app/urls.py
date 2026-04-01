##########################################################################
##                                                                      ##
##  Copyright (c) 2019 Philipp Lösel. All rights reserved.              ##
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

from django.urls import re_path
from . import views
from . import repository
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views

urlpatterns = [
    # biomedisa
    re_path(r'^$', views.index, name='index'),
    # repository
    re_path(r'^repository/(?P<id>\d+)/$', repository.repository, name='repository'),
    re_path(r'^repository/specimen/(?P<id>\d+)/$', repository.specimen_info, name='specimen_info'),
    re_path(r'^repository/data/(?P<id>\d+)/$', repository.tomographic_info, name='tomographic_info'),
    re_path(r'^repository/sliceviewer/$', repository.sliceviewer_repository, name='sliceviewer_repository'),
    re_path(r'^repository/visualization/$', repository.visualization_repository, name='visualization_repository'),
    re_path(r'^repository/download/$', repository.download_repository, name='download_repository'),
    re_path(r'^repository/share_data/$', repository.share_repository_data, name='share_repository_data'),
    # manage repository
    re_path(r'^repository/unsubscribe/(?P<id>\d+)/$', repository.unsubscribe_from_repository, name='unsubscribe_from_repository'),
    re_path(r'^share_repository/$', repository.share_repository, name='share_repository'),
    # antscan
    re_path(r'^antscan/$', repository.repository, name='repository'),
    re_path(r'^antscan/specimen/(?P<id>\d+)/$', repository.specimen_info, name='specimen_info'),
    re_path(r'^antscan/data/(?P<id>\d+)/$', repository.tomographic_info, name='tomographic_info'),
    re_path(r'^antscan/sliceviewer/$', repository.sliceviewer_repository, name='sliceviewer_repository'),
    re_path(r'^antscan/visualization/$', repository.visualization_repository, name='visualization_repository'),
    re_path(r'^antscan/download/$', repository.download_repository, name='download_repository'),
    re_path(r'^antscan/share_data/$', repository.share_repository_data, name='share_repository_data'),
    # app
    re_path(r'^app/$', views.app, name='app'),
    # demo
    re_path(r'^paraview/$', views.paraview, name='paraview'),
    # gallery
    re_path(r'^quartz/$', views.file_list, name="file_list"),
    re_path(r'^gallery/$', views.gallery, name='gallery'),
    # contact
    re_path(r'^contact/$', views.contact, name='contact'),
    # impressum
    re_path(r'^impressum/$', views.impressum, name='impressum'),
    # partners
    re_path(r'^partners/$', views.partners, name='partners'),
    # tutorial
    re_path(r'^faq/$', views.faq, name='faq'),
    # login
    re_path(r'^login/$', views.login_user, name='login'),
    re_path(r'^password/$', views.change_password, name='change_password'),
    # user profile
    re_path(r'^profile/$', views.update_profile, name='update_profile'),
    # logout
    re_path(r'^logout/$', views.logout_user, name='logout'),
    # delete files
    re_path(r'^delete/$', views.delete, name='delete'),
    re_path(r'^delete/demo/$', views.delete_demo, name='delete_demo'),
    # delete account
    re_path(r'^delete_account/$', views.delete_account, name='delete_account'),
    # download files
    re_path(r'^download/(?P<id>\d+)/$', views.download, name='download'),
    re_path(r'^download/demo/$', views.download_demo, name='download_demo'),
    # run segmentation
    re_path(r'^run/$', views.run, name='run'),
    re_path(r'^run/demo/$', views.run_demo, name='run_demo'),
    # visualization
    re_path(r'^visualization/$', views.visualization, name='visualization'),
    re_path(r'^visualization_demo/$', views.visualization_demo, name='visualization_demo'),
    # sliceviewer
    re_path(r'^imageviewer/(?P<id>\d+)/$', views.imageviewer, name='imageviewer'),
    re_path(r'^sliceviewer/(?P<id>\d+)/$', views.sliceviewer, name='sliceviewer'),
    re_path(r'^sliceviewer_demo/$', views.sliceviewer_demo, name='sliceviewer_demo'),
    # stop running process or remove from queue
    re_path(r'^remove_from_queue/$', views.remove_from_queue, name='remove_from_queue'),
    # reset a file
    re_path(r'^reset/(?P<id>\d+)/$', views.reset, name='reset'),
    # create account
    re_path(r'^register/$', views.register, name='register'),
    re_path(r'^activate/(?P<key>.+)$', views.activation, name='activation'),
    # settings
    re_path(r'^settings/(?P<id>\d+)/$', views.settings, name='settings'),
    re_path(r'^settings/prediction/(?P<id>\d+)/$', views.settings_prediction, name='settings_prediction'),
    # storage
    re_path(r'^storage/$', views.storage, name='storage_up'),
    # move
    re_path(r'^move/$', views.move, name='move'),
    re_path(r'^rename/$', views.rename_file, name='rename_file'),
    # reactivate
    re_path(r'^reactivate/$', views.reactivate_file, name='reactivate_file'),
    # share data
    re_path(r'^share/$', views.share_data, name='share_data'),
    re_path(r'^create/(?P<id>\d+)/$', views.create_download_link, name='create_download_link'),
    re_path(r'^accept/$', views.accept_shared_data, name='accept_shared_data'),
    re_path(r'^download/shared/(?P<id>\d+)/(?P<pw>.*)/$', views.download_shared_data, name='download_shared_data'),
    # clean state
    re_path(r'^cleanstate/(?P<next>\w+)/$', views.clean_state, name='clean_state'),
    # change active final
    re_path(r'^change_active_final/(?P<id>\d+)/(?P<val>\d+)/$', views.change_active_final, name='change_active_final'),
    # check status
    re_path(r'^status/$', views.status, name='status'),
    # dummy
    re_path(r'^dummy/$', views.dummy, name='dummy'),
    # password reset
    re_path(r'^password_reset/$', auth_views.PasswordResetView.as_view(template_name='registration/password_reset_form.html'), name='password_reset'),
    re_path(r'^password_reset/done/$', auth_views.PasswordResetDoneView.as_view(template_name='registration/password_reset_done.html'), name='password_reset_done'),
    re_path(r'^reset/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,6}-[0-9A-Za-z]{1,32})/$',
        auth_views.PasswordResetConfirmView.as_view(template_name='registration/password_reset_confirm.html'), name='password_reset_confirm'),
    re_path(r'^reset/done/$', auth_views.PasswordResetCompleteView.as_view(template_name='registration/password_reset_complete.html'), name='password_reset_complete'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) + static(settings.PARAVIEW_URL, document_root=settings.PARAVIEW_ROOT)

