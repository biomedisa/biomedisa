# Generated by Django 3.0.4 on 2020-03-22 13:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('biomedisa_app', '0055_auto_20200321_1543'),
    ]

    operations = [
        migrations.AddField(
            model_name='mushroomspot',
            name='date',
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
        migrations.AddField(
            model_name='mushroomspot',
            name='status',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='mushroomspot',
            name='status_pic',
            field=models.ImageField(null=True, upload_to=''),
        ),
        migrations.AlterField(
            model_name='mushroomspot',
            name='picture',
            field=models.ImageField(null=True, upload_to=''),
        ),
    ]
