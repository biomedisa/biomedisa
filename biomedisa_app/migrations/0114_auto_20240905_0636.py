# Generated by Django 3.2.6 on 2024-09-05 04:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('biomedisa_app', '0113_rename_name_recommended_specimen_name'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='repository',
            name='featured_img_height',
        ),
        migrations.RemoveField(
            model_name='repository',
            name='featured_img_width',
        ),
        migrations.AddField(
            model_name='repository',
            name='featured_url',
            field=models.TextField(null=True),
        ),
    ]
