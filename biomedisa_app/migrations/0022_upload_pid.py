# -*- coding: utf-8 -*-
# Generated by Django 1.11.15 on 2018-09-28 16:53
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('biomedisa_app', '0021_auto_20180522_1523'),
    ]

    operations = [
        migrations.AddField(
            model_name='upload',
            name='pid',
            field=models.IntegerField(default=0),
        ),
    ]
