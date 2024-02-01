[![biomedisa](biomedisa_app/static/biomedisa_logo.svg)](https://biomedisa.info)
-----------
- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Installation (command-line based](#installation-command-line-based)
- [Installation (browser based)](#installation-browser-based)
- [Download Data](#download-data)
- [Smart Interpolation](#smart-interpolation)
- [Deep Learning](#deep-learning)
- [Biomedisa Features](#biomedisa-features)
- [Update Biomedisa](#update-biomedisa)
- [Releases](#releases)
- [Authors](#authors)
- [FAQ](#faq)
- [Citation](#citation)
- [License](#license)

# Overview
Biomedisa (https://biomedisa.info) is a free and easy-to-use open-source application for segmenting large volumetric images, e.g. CT and MRI scans, developed at The Australian National University CTLab. Biomedisa's semi-automated segmentation is based on a smart interpolation of sparsely pre-segmented slices, taking into account the complete underlying image data. In addition, Biomedisa enables deep learning for the fully automated segmentation of series of similar samples. It can be used in combination with segmentation tools such as Amira/Avizo, ImageJ/Fiji and 3D Slicer. If you are using Biomedisa or the data for your research please cite: Lösel, P.D. et al. [Introducing Biomedisa as an open-source online platform for biomedical image segmentation.](https://www.nature.com/articles/s41467-020-19303-w) *Nat. Commun.* **11**, 5577 (2020).

# Hardware Requirements
+ One or more NVIDIA GPUs with compute capability 3.0 or higher or an Intel CPU.
+ Depending on the size of the image data (32 GB or more).

# Installation (command-line based)
+ [Ubuntu 22.04 + CUDA + GPU (recommended)](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2204_cuda11.8_gpu_cli.md)
+ [Ubuntu 22.04 + OpenCL + CPU (smart interpolation only and very slow)](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2204_opencl_cpu_cli.md)
+ [Windows 10 + CUDA + GPU (recommended)](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_cuda_gpu_cli.md)
+ [Windows 10 + OpenCL + GPU (easy to install but lacks features like allaxis, smoothing, uncertainty, optimized GPU memory usage)](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_opencl_gpu_cli.md)
+ [Windows 10 + OpenCL + CPU (very slow)](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_opencl_cpu_cli.md)

# Installation (browser based)
+ [Ubuntu 22.04](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2204_cuda11.8.md)

# Download Data
+ Download the data from our [gallery](https://biomedisa.info/gallery/)

# Smart Interpolation
+ [Parameters and Examples](https://github.com/biomedisa/biomedisa/blob/master/README/smart_interpolation.md)

#### Python example
```python
# change this line to your biomedisa directory
path_to_biomedisa = '/home/<user>/git/biomedisa'

import sys
sys.path.append(path_to_biomedisa)
from biomedisa_features.biomedisa_helper import load_data, save_data
from biomedisa_features.biomedisa_interpolation import smart_interpolation

# load data
img, _ = load_data('Downloads/trigonopterus.tif')
labels, header = load_data('Downloads/labels.trigonopterus_smart.am')

# run smart interpolation with optional smoothing result
results = smart_interpolation(img, labels, smooth=100)

# get results
regular_result = results['regular']
smooth_result = results['smooth']

# save results
save_data('Downloads/final.trigonopterus.am', regular_result, header=header)
save_data('Downloads/final.trigonopterus.smooth.am', smooth_result, header=header)
```

#### Command-line based
```
# change to the features directory
cd git/biomedisa/biomedisa_features/

# start smart interpolation
python biomedisa_interpolation.py C:\Users\%USERNAME%\Downloads\tumor.tif C:\Users\%USERNAME%\Downloads\labels.tumor.tif
```

# Deep Learning
+ [Parameters and Examples](https://github.com/biomedisa/biomedisa/blob/master/README/deep_learning.md)

#### Python example (training)
```python
# change this line to your biomedisa directory
path_to_biomedisa = '/home/<user>/git/biomedisa'

# load libraries
import sys
sys.path.append(path_to_biomedisa)
from biomedisa_features.biomedisa_helper import load_data
from biomedisa_features.biomedisa_deeplearning import deep_learning

# load image data
img1, _ = load_data('Head1.am')
img2, _ = load_data('Head2.am')
img_data = [img1, img2]

# load label data
label1, _ = load_data('Head1.labels.am')
label2, header, ext = load_data('Head2.labels.am',
        return_extension=True)
label_data = [label1, label2]

# load validation data (optional)
img3, _ = load_data('Head3.am')
img4, _ = load_data('Head4.am')
label3, _ = load_data('Head3.labels.am')
label4, _ = load_data('Head4.labels.am')
val_img_data = [img3, img4]
val_label_data = [label3, label4]

# deep learning 
deep_learning(img_data, label_data, train=True, batch_size=12,
        val_img_data=val_img_data, val_label_data=val_label_data,
        header=header, extension=ext, path_to_model='honeybees.h5')
```

#### Command-line based (training)
```
# change to the features directory
cd git/biomedisa/biomedisa_features/

# start training with a batch size of 12
python biomedisa_deeplearning.py C:\Users\%USERNAME%\Downloads\training_heart C:\Users\%USERNAME%\Downloads\training_heart_labels -t -bs 12

# validation (optional)
python biomedisa_deeplearning.py C:\Users\%USERNAME%\Downloads\training_heart C:\Users\%USERNAME%\Downloads\training_heart_labels -t -vi C:\Users\%USERNAME%\Downloads\val_img -vl C:\Users\%USERNAME%\Downloads\val_labels
```
If running into ResourceExhaustedError due to out of memory (OOM), try to use smaller batch size.

#### Python example (prediction)
```python
# change this line to your biomedisa directory
path_to_biomedisa = '/home/<user>/git/biomedisa'

# load libraries
import sys
sys.path.append(path_to_biomedisa)
from biomedisa_features.biomedisa_helper import load_data, save_data
from biomedisa_features.biomedisa_deeplearning import deep_learning

# load data
img, _ = load_data('Head5.am')

# deep learning
results = deep_learning(img, predict=True,
        path_to_model='honeybees.h5', batch_size=6)

# save result
save_data('final.Head5.am', results['regular'])
```

#### Command-line based (prediction)
```
# change to the features directory
cd git/biomedisa/biomedisa_features/

# start prediction with a batch size of 6
python biomedisa_deeplearning.py C:\Users\%USERNAME%\Downloads\testing_axial_crop_pat13.nii.gz C:\Users\%USERNAME%\Downloads\heart.h5 -p -bs 6
```

# Biomedisa Features

#### Load and save data (such as Amira Mesh, TIFF, NRRD, NIfTI or DICOM)
```python
import sys
sys.path.append(path_to_biomedisa)  # e.g. '/home/<user>/git/biomedisa'
from biomedisa_features.biomedisa_helper import load_data, save_data

# load data as numpy array (for DICOM, path_to_data must be a directory containing the slices)
data, header = load_data(path_to_data)

# save data (for TIFF, header=None)
save_data(path_to_data, data, header)
```

#### Create STL mesh from segmentation (label values are saved as attributes)
```python
import os, sys
sys.path.append(path_to_biomedisa)  # e.g. '/home/<user>/git/biomedisa'
from biomedisa_features.biomedisa_helper import load_data, save_data
from biomedisa_features.create_mesh import get_voxel_spacing, save_mesh

# load segmentation
data, header, extension = load_data(path_to_data, return_extension=True)

# get voxel spacing
x_res, y_res, z_res = get_voxel_spacing(header, data, extension)
print(f'Voxel spacing: x_spacing, y_spacing, z_spacing = {x_res}, {y_res}, {z_res}')

# save stl file
path_to_data = path_to_data.replace(os.path.splitext(path_to_data)[1],'.stl')
save_mesh(path_to_data, data, x_res, y_res, z_res, poly_reduction=0.9, smoothing_iterations=15)
```

#### Create mesh directly
```
python git/biomedisa/biomedisa_features/create_mesh.py <path_to_data>
```

#### Options
`--poly_reduction` or `-pr`: Reduce number of polygons by this factor (default: 0.9)

`--smoothing_iterations` or `-s`: Iteration steps for smoothing (default: 15)

`--x_res` or `-xres`: Voxel spacing/resolution x-axis (default: None)

`--y_res` or `-yres`: Voxel spacing/resolution y-axis (default: None)

`--z_res` or `-zres`: Voxel spacing/resolution z-axis (default: None)

#### Resize data
```python
import os, sys
sys.path.append(path_to_biomedisa)  # e.g. '/home/<user>/git/biomedisa'
from biomedisa_features.biomedisa_helper import img_resize

# resize image data
zsh, ysh, xsh = data.shape
new_zsh, new_ysh, new_xsh = zsh//2, ysh//2, xsh//2
data = img_resize(data, new_zsh, new_ysh, new_xsh)

# resize label data
label_data = img_resize(label_data, new_zsh, new_ysh, new_xsh, labels=True)
```

#### Remove outliers and fill holes
```python
from biomedisa_features.biomedisa_helper import clean, fill

# delete outliers smaller than 90% of the segment
label_data = clean(label_data, 0.9)

# fill holes
label_data = fill(label_data, 0.9)
```

#### Accuracy assessment
```python
from biomedisa_features.biomedisa_helper import Dice_score, ASSD
dice = Dice_score(ground_truth, result)
assd = ASSD(ground_truth, result)
```

# Update Biomedisa
If you have used `git clone`, change to the Biomedisa directory and make a pull request.
```
cd git/biomedisa
git pull
```

If you have installed the full version of Biomedisa (including MySQL database), you also need to update the database.
```
python manage.py migrate
```

If you have installed an [Apache Server](https://github.com/biomedisa/biomedisa/blob/master/README/APACHE_SERVER.md), you need to restart the server.
```
sudo service apache2 restart
```

# Releases

For the versions available, see the [list of releases](https://github.com/biomedisa/biomedisa/releases). 

# Authors

* **Philipp D. Lösel**

See also the list of [contributors](https://github.com/biomedisa/biomedisa/blob/master/credits.md) who participated in this project.

# FAQ
Frequently asked questions can be found at: https://biomedisa.info/faq/.

# Citation

If you use the package or the online platform, please cite the following paper.

`Lösel, P.D. et al. Introducing Biomedisa as an open-source online platform for biomedical image segmentation. Nat. Commun. 11, 5577 (2020).`

# License

This project is covered under the **EUROPEAN UNION PUBLIC LICENCE v. 1.2 (EUPL)**.

