# Generated by Django 2.1.5 on 2019-03-22 15:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('biomedisa_app', '0038_auto_20190322_1457'),
    ]

    operations = [
        migrations.AlterField(
            model_name='profile',
            name='activation_key',
            field=models.TextField(null=True),
        ),
    ]
