# Generated by Django 3.2.6 on 2023-11-30 00:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('biomedisa_app', '0103_upload_validation_data'),
    ]

    operations = [
        migrations.AlterField(
            model_name='profile',
            name='storage_size',
            field=models.IntegerField(default=1000),
        ),
    ]