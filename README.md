# Biomedisa

- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation (command-line-only)](#installation-command-line-only)
- [Installation (browser based)](#installation-browser-based)
- [Data](#data)
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
+ Ddepending on the size of the image data (e.g. 32 GB).

# Software Requirements
+ [NVIDIA GPU drivers](https://www.nvidia.com/drivers) for GPU support
+ [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) or [Intel Runtime for OpenCL](https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html)

# Installation (command-line-only)
+ [Ubuntu 22.04 + Smart Interpolation + CUDA + GPU](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2204_interpolation_cuda11.8_gpu_cli.md)
+ [Ubuntu 22.04 + Smart Interpolation + OpenCL + CPU](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2204_interpolation_opencl_cpu_cli.md)
+ [Ubuntu 22.04 + Deep Learning](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2204_deeplearning_cuda11.8_cli.md)
+ [Windows 10 + Smart Interpolation + CUDA + GPU](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_interpolation_cuda_gpu_cli.md)
+ [Windows 10 + Smart Interpolation + OpenCL + GPU](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_interpolation_opencl_gpu_cli.md)
+ [Windows 10 + Smart Interpolation + OpenCL + CPU](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_interpolation_opencl_cpu_cli.md)
+ [Windows 10 + Deep Learning](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_deeplearning_cuda11.3_cli.md)

# Installation (browser based)
+ [Ubuntu 22.04](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2204_cuda11.8.md)
+ [Windows 10 (21H2 or higher)](https://github.com/biomedisa/biomedisa/blob/master/README/windows11.md)
+ [Windows 11](https://github.com/biomedisa/biomedisa/blob/master/README/windows11.md)

# Data
+ Download data from the [gallery](https://biomedisa.org/gallery/)

# Smart Interpolation
+ [Parameters](https://github.com/biomedisa/biomedisa/blob/master/README/smart_interpolation.md)

#### Python example
```
import sys
sys.path.append(path_to_biomedisa)  # e.g. '/home/<user>/git/biomedisa'
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

#### Automatic segmentation using a trained network and batch size 6
```
# change to the demo directory
cd ~/git/biomedisa/demo/

# Ubuntu
python3 biomedisa_deeplearning.py ~/Downloads/testing_axial_crop_pat13.nii.gz ~/Downloads/heart.h5 -p -bs 6

# Windows
python biomedisa_deeplearning.py Downloads\testing_axial_crop_pat13.nii.gz Downloads\heart.h5 -p -bs 6
```

#### Train a neural network for automatic segmentation
```
# change to the demo directory
cd ~/git/biomedisa/demo/

# Ubuntu
python3 biomedisa_deeplearning.py ~/Downloads/training_heart ~/Downloads/training_heart_labels -t

# Windows
python biomedisa_deeplearning.py Downloads\training_heart Downloads\training_heart_labels -t
```

#### Options
`--help` or `-h`: show more information and exit

`--version` or `-v`: Biomedisa version

`--predict` or `-p`: automatic/predict segmentation

`--train` or `-t`: train a neural network

`--epochs INT` or `-e INT`: number of epochs trained (default: 100)

`--batch_size INT` or `-bs INT`: batch size (default: 24). If you have a memory error, try reducing to 6, for example.

`--val_images PATH` or `-vi PATH`: path to directory with validation images

`--val_labels PATH` or `-vl PATH`: path to directory with validation labels

`--validation_split FLOAT` or `-vs FLOAT`: for example, split your data into 80% training data and 20% validation data with `-vs 0.8`

`--early_stopping INT` or `-es INT`: stop training if there is no improvement after specified number of epochs

`--no_compression` or `-nc`: disable compression of segmentation results (default: False)

`--create_slices` or `-cs`: create slices of segmentation results (default: False)

`--ignore STR`: ignore specific label(s), e.g. "2,5,6" (default: none)

`--only STR`: segment only specific label(s), e.g. "1,3,5" (default: all)

`--clean FLOAT` or `-c FLOAT`: remove outliers, e.g. 0.5 means that objects smaller than 50 percent of the size of the largest object will be removed (default: None)

`--fill FLOAT` or `-f FLOAT`: fill holes, e.g. 0.5 means that all holes smaller than 50 percent of the entire label will be filled (default: None)

`--balance` or `-b`: Balance foreground and background training patches (default: False)

`--flip_x`: Randomly flip x-axis during training (default: False)

`--flip_y`: Randomly flip y-axis during training (default: False)

`--flip_z`: Randomly flip z-axis during training (default: False)

`--network_filters STR` or `-nf STR`: Number of filters per layer up to the deepest, e.g. "32-64-128-256-512" (default: "32-64-128-256-512")

`--resnet` or `-rn`: Use U-resnet instead of standard U-net (default: False)

`--no_normalization` or `-nn`: Disable image normalization (default: False)

`--rotate FLOAT` or `-r FLOAT`: Randomly rotate during training (default: 0.0)

`--learning_rate FLOAT` or `-lr`: Learning rate (default: 0.01)

`--stride_size [1-64]` or `-ss [1-64]`: Stride size for patches (default: 32)

`--validation_stride_size [1-64]` or `-vss [1-64]`: Stride size for validation patches (default: 32)

`--validation_freq INT` or `-vf INT`: Epochs performed before validation (default: 1)

`--validation_batch_size INT` or `-vbs INT`: validation batch size (default: 24)

`--x_scale INT` or `-xs INT`: Images and labels are scaled at x-axis to this size before training (default: 256)

`--y_scale INT` or `-ys INT`: Images and labels are scaled at y-axis to this size before training (default: 256)

`--z_scale INT` or `-zs INT`: Images and labels are scaled at z-axis to this size before training (default: 256)

`--no_scaling` or `-ns`: Do not resize image and label data (default: False)

#### Accuracy Assessment: Dice Score vs. Standard Accuracy in Biomedisa
`--val_tf` or `-vt`: use standard pixelwise accuracy provided by TensorFlow (default: False). When evaluating accuracy, Biomedisa relies on the Dice score rather than the standard accuracy. The Dice score offers a more reliable assessment by measuring the overlap between the segmented regions, whereas the standard accuracy also considers background classification, which can lead to misleading results, especially when dealing with small segments within a much larger volume. Even if half of the segment is mislabeled, the standard accuracy may still yield a remarkably high value. However, if you still prefer to use the standard accuracy, you can enable it by using this option.

#### Automatic cropping
`--crop_data` or `-cd`: Both the training and inference data should be cropped to the region of interest for best performance. As an alternative to manual cropping, you can use Biomedisa's AI-based automatic cropping. After training, auto cropping is automatically applied to your inference data.

`--save_cropped` or `-sc`: save cropped image (default: False)

# Biomedisa Features

#### Load and save data (such as Amira Mesh, TIFF, NRRD, NIfTI or DICOM)
```
import sys
sys.path.append(path_to_biomedisa)  # e.g. '/home/<user>/git/biomedisa'
from biomedisa_features.biomedisa_helper import load_data, save_data

# load data as numpy array (for DICOM, path_to_data must be a directory containing the slices)
data, header = load_data(path_to_data)

# save data (for TIFF, header=None)
save_data(path_to_data, data, header)
```

#### Create STL mesh from segmentation (label values are saved as attributes)
```
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
```
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
```
from biomedisa_features.biomedisa_helper import clean, fill

# delete outliers smaller than 90% of the segment
label_data = clean(label_data, 0.9)

# fill holes
label_data = fill(label_data, 0.9)
```

#### Measure accuracy
```
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

