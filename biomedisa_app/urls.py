##########################################################################
##                                                                      ##
##  Copyright (c) 2024 Philipp Lösel. All rights reserved.              ##
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

from django.conf.urls import url
from . import views
from . import repository
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views

urlpatterns = [
    # biomedisa
    url(r'^$', views.index, name='index'),
    # repository
    url(r'^repository/(?P<id>\d+)/$', repository.repository, name='repository'),
    url(r'^repository/specimen/(?P<id>\d+)/$', repository.specimen_info, name='specimen_info'),
    url(r'^repository/data/(?P<id>\d+)/$', repository.tomographic_info, name='tomographic_info'),
    url(r'^repository/sliceviewer/$', repository.sliceviewer_repository, name='sliceviewer_repository'),
    url(r'^repository/visualization/$', repository.visualization_repository, name='visualization_repository'),
    url(r'^repository/download/$', repository.download_repository, name='download_repository'),
    url(r'^repository/share_data/$', repository.share_repository_data, name='share_repository_data'),
    # manage repository
    url(r'^repository/unsubscribe/(?P<id>\d+)/$', repository.unsubscribe_from_repository, name='unsubscribe_from_repository'),
    url(r'^share_repository/$', repository.share_repository, name='share_repository'),
    # antscan
    url(r'^antscan/$', repository.repository, name='repository'),
    url(r'^antscan/specimen/(?P<id>\d+)/$', repository.specimen_info, name='specimen_info'),
    url(r'^antscan/data/(?P<id>\d+)/$', repository.tomographic_info, name='tomographic_info'),
    url(r'^antscan/sliceviewer/$', repository.sliceviewer_repository, name='sliceviewer_repository'),
    url(r'^antscan/visualization/$', repository.visualization_repository, name='visualization_repository'),
    url(r'^antscan/download/$', repository.download_repository, name='download_repository'),
    url(r'^antscan/share_data/$', repository.share_repository_data, name='share_repository_data'),
    # app
    url(r'^app/$', views.app, name='app'),
    # demo
    url(r'^paraview/$', views.paraview, name='paraview'),
    # demo
    url(r'^gallery/$', views.gallery, name='gallery'),
    # contact
    url(r'^contact/$', views.contact, name='contact'),
    # impressum
    url(r'^impressum/$', views.impressum, name='impressum'),
    # partners
    url(r'^partners/$', views.partners, name='partners'),
    # tutorial
    url(r'^faq/$', views.faq, name='faq'),
    # login
    url(r'^login/$', views.login_user, name='login'),
    url(r'^password/$', views.change_password, name='change_password'),
    # user profile
    url(r'^profile/$', views.update_profile, name='update_profile'),
    # logout
    url(r'^logout/$', views.logout_user, name='logout'),
    # delete files
    url(r'^delete/$', views.delete, name='delete'),
    url(r'^delete/demo/$', views.delete_demo, name='delete_demo'),
    # delete account
    url(r'^delete_account/$', views.delete_account, name='delete_account'),
    # download files
    url(r'^download/(?P<id>\d+)/$', views.download, name='download'),
    url(r'^download/demo/$', views.download_demo, name='download_demo'),
    # run segmentation
    url(r'^run/$', views.run, name='run'),
    url(r'^run/demo/$', views.run_demo, name='run_demo'),
    # visualization
    url(r'^visualization/$', views.visualization, name='visualization'),
    url(r'^visualization_demo/$', views.visualization_demo, name='visualization_demo'),
    # sliceviewer
    url(r'^imageviewer/(?P<id>\d+)/$', views.imageviewer, name='imageviewer'),
    url(r'^sliceviewer/(?P<id>\d+)/$', views.sliceviewer, name='sliceviewer'),
    url(r'^sliceviewer_demo/$', views.sliceviewer_demo, name='sliceviewer_demo'),
    # stop running process or remove from queue
    url(r'^remove_from_queue/$', views.remove_from_queue, name='remove_from_queue'),
    # reset a file
    url(r'^reset/(?P<id>\d+)/$', views.reset, name='reset'),
    # create account
    url(r'^register/$', views.register, name='register'),
    url(r'^activate/(?P<key>.+)$', views.activation, name='activation'),
    # settings
    url(r'^settings/(?P<id>\d+)/$', views.settings, name='settings'),
    url(r'^settings/prediction/(?P<id>\d+)/$', views.settings_prediction, name='settings_prediction'),
    # storage
    url(r'^storage/$', views.storage, name='storage_up'),
    # move
    url(r'^move/$', views.move, name='move'),
    url(r'^rename/$', views.rename_file, name='rename_file'),
    # reactivate
    url(r'^reactivate/$', views.reactivate_file, name='reactivate_file'),
    # share data
    url(r'^share/$', views.share_data, name='share_data'),
    url(r'^create/(?P<id>\d+)/$', views.create_download_link, name='create_download_link'),
    url(r'^accept/$', views.accept_shared_data, name='accept_shared_data'),
    url(r'^download/shared/(?P<id>\d+)/(?P<pw>.*)/$', views.download_shared_data, name='download_shared_data'),
    # clean state
    url(r'^cleanstate/(?P<next>\w+)/$', views.clean_state, name='clean_state'),
    # change active final
    url(r'^change_active_final/(?P<id>\d+)/(?P<val>\d+)/$', views.change_active_final, name='change_active_final'),
    # check status
    url(r'^status/$', views.status, name='status'),
    # dummy
    url(r'^dummy/$', views.dummy, name='dummy'),
    # password reset
    url(r'^password_reset/$', auth_views.PasswordResetView.as_view(template_name='registration/password_reset_form.html'), name='password_reset'),
    url(r'^password_reset/done/$', auth_views.PasswordResetDoneView.as_view(template_name='registration/password_reset_done.html'), name='password_reset_done'),
    url(r'^reset/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>[0-9A-Za-z]{1,6}-[0-9A-Za-z]{1,32})/$',
        auth_views.PasswordResetConfirmView.as_view(template_name='registration/password_reset_confirm.html'), name='password_reset_confirm'),
    url(r'^reset/done/$', auth_views.PasswordResetCompleteView.as_view(template_name='registration/password_reset_complete.html'), name='password_reset_complete'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) + static(settings.PARAVIEW_URL, document_root=settings.PARAVIEW_ROOT)

