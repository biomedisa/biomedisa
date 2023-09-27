# Biomedisa

- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Installation (command-line-only)](#installation-command-line-only)
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
Biomedisa (https://biomedisa.org) is a free and easy-to-use open-source online platform for segmenting large volumetric images, e.g. CT and MRI scans, at Heidelberg University and the Australian National University. Biomedisa's semi-automated segmentation is based on a smart interpolation of sparsely pre-segmented slices, taking into account the complete underlying image data. In addition, Biomedisa enables deep learning for the fully automated segmentation of series of similar samples. It can be used in combination with segmentation tools such as Amira/Avizo, ImageJ/Fiji and 3D Slicer. If you are using Biomedisa or the data for your research please cite: Lösel, P.D. et al. [Introducing Biomedisa as an open-source online platform for biomedical image segmentation.](https://www.nature.com/articles/s41467-020-19303-w) *Nat. Commun.* **11**, 5577 (2020).

# Hardware Requirements
+ One or more NVIDIA GPUs with compute capability 3.0 or higher or an Intel CPU.
+ Depending on the size of the image data (32 GB or more).

# Installation (command-line-only)
+ [Ubuntu 22.04 + Smart Interpolation + CUDA + GPU (recommended)](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2204_interpolation_cuda11.8_gpu_cli.md)
+ [Ubuntu 22.04 + Smart Interpolation + OpenCL + CPU](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2204_interpolation_opencl_cpu_cli.md)
+ [Ubuntu 22.04 + Deep Learning](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2204_deeplearning_cuda11.8_cli.md)
+ [Windows 10 + Smart Interpolation + CUDA + GPU (recommended)](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_interpolation_cuda_gpu_cli.md)
+ [Windows 10 + Smart Interpolation + OpenCL + GPU (easy to install but lacks features like allaxis and smoothing](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_interpolation_opencl_gpu_cli.md)
+ [Windows 10 + Smart Interpolation + OpenCL + CPU](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_interpolation_opencl_cpu_cli.md)
+ [Windows 10 + Deep Learning](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_deeplearning_cuda11.3_cli.md)

# Installation (browser based)
+ [Ubuntu 22.04](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2204_cuda11.8.md)
+ [Windows](https://github.com/biomedisa/biomedisa/blob/master/README/windows11.md)

# Download Data
+ Download the data from our [gallery](https://biomedisa.org/gallery/)

# Smart Interpolation
+ [Parameters](https://github.com/biomedisa/biomedisa/blob/master/README/smart_interpolation.md)

#### Python example
```python
# change this line to your biomedisa directory
path_to_biomedisa = '/home/<user>/git/biomedisa'

import sys
sys.path.append(path_to_biomedisa)
from biomedisa_features.biomedisa_helper import load_data, save_data
from demo.biomedisa_interpolation import smart_interpolation

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
# Ubuntu
python3 ~/git/biomedisa/demo/biomedisa_interpolation.py ~/Downloads/tumor.tif ~/Downloads/labels.tumor.tif

# Windows
python git\biomedisa\demo\biomedisa_interpolation.py Downloads\tumor.tif Downloads\labels.tumor.tif
```

#### Multi-GPU (e.g. 4 GPUs)
```
# Ubuntu
mpiexec -np 4 python3 ~/git/biomedisa/demo/biomedisa_interpolation.py ~/Downloads/NMB_F2875.tif ~/Downloads/labels.NMB_F2875.tif

# Windows
mpiexec -np 4 python -u git\biomedisa\demo\biomedisa_interpolation.py Downloads\NMB_F2875.tif Downloads\labels.NMB_F2875.tif
```

#### Memory error
If memory errors (either GPU or host memory) occur, you can start the segmentation as follows:
```
python3 ~/git/biomedisa/demo/split_volume.py 'path_to_image' 'path_to_labels' -np 4 -sz 2 -sy 2 -sx 2
```
Where `-n` is the number of GPUs and each axis (`x`,`y` and `z`) is divided into two overlapping parts. The volume is thus divided into `2*2*2=8` subvolumes. These are segmented separately and then reassembled.

# Deep Learning
+ [Parameters](https://github.com/biomedisa/biomedisa/blob/master/README/deep_learning.md)

#### Train a neural network for automatic segmentation
```
# change to the demo directory
cd ~/git/biomedisa/demo/

# Ubuntu
python3 biomedisa_deeplearning.py ~/Downloads/training_heart ~/Downloads/training_heart_labels -t

# Windows
python biomedisa_deeplearning.py Downloads\training_heart Downloads\training_heart_labels -t
```

#### Python example (training)
```python
# change this line to your biomedisa directory
path_to_biomedisa = '/home/<user>/git/biomedisa'

# load libraries
import sys
sys.path.append(path_to_biomedisa)
from biomedisa_features.biomedisa_helper import load_data
from demo.biomedisa_deeplearning import deep_learning

# load image data
img1, _ = load_data('Head1.am')
img2, _ = load_data('Head2.am')
img_list = [img1, img2]

# load label data
label1, _ = load_data('Head1.labels.am')
label2, header, ext = load_data('Head2.labels.am',
        return_extension=True)
label_list = [label1, label2]

# deep learning
deep_learning(img_list, label_list, train=True, batch_size=12,
        header=header, extension=ext, path_to_model='honeybees.h5')
```

#### Automatic segmentation using a trained network and a batch size of 6
```
# change to the demo directory
cd ~/git/biomedisa/demo/

# Ubuntu
python3 biomedisa_deeplearning.py ~/Downloads/testing_axial_crop_pat13.nii.gz ~/Downloads/heart.h5 -p -bs 6

# Windows
python biomedisa_deeplearning.py Downloads\testing_axial_crop_pat13.nii.gz Downloads\heart.h5 -p -bs 6
```

#### Python example (prediction)
```python
# change this line to your biomedisa directory
path_to_biomedisa = '/home/<user>/git/biomedisa'

# load libraries
import sys
sys.path.append(path_to_biomedisa)
from biomedisa_features.biomedisa_helper import load_data, save_data
from demo.biomedisa_deeplearning import deep_learning
from demo.keras_helper import get_image_dimensions, get_physical_size

# load data
img, img_header, img_ext = load_data('Head3.am',
        return_extension=True)

# deep learning
results = deep_learning(img, predict=True, img_header=img_header,
        path_to_model='honeybees.h5', img_extension=img_ext)

# save result
save_data('final.Head3.am', results['regular'],
        header=results['header'])
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
python3 git/biomedisa/biomedisa_features/create_mesh.py <path_to_data>
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

#### Measure accuracy
```python
from biomedisa_features.helper import Dice_score, ASSD

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
python3 manage.py migrate
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
Frequently asked questions can be found at: https://biomedisa.org/faq/.

# Citation

If you use the package or the online platform, please cite the following paper.

`Lösel, P.D. et al. Introducing Biomedisa as an open-source online platform for biomedical image segmentation. Nat. Commun. 11, 5577 (2020).`

# License

This project is covered under the **EUROPEAN UNION PUBLIC LICENCE v. 1.2 (EUPL)**.

