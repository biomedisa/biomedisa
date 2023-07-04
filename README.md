# Biomedisa

- [Overview](#overview)
- [Hardware requirements](#hardware-requirements)
- [Software requirements](#software-requirements)
- [Installation (command-line-only)](#installation-command-line-only)
- [Full installation (GUI)](#full-installation-gui)
- [Biomedisa interpolation](#biomedisa-interpolation)
- [Biomedisa deep learning](#biomedisa-deep-learning)
- [Biomedisa features](#biomedisa-features)
- [Update Biomedisa](#update-biomedisa)
- [Releases](#releases)
- [Authors](#authors)
- [FAQ](#faq)
- [Citation](#citation)
- [License](#license)

# Overview
Biomedisa (https://biomedisa.org) is a free and easy-to-use open-source online platform for segmenting large volumetric images, e.g. CT and MRI scans, at Heidelberg University and the Australian National University. Biomedisa's semi-automated segmentation is based on a smart interpolation of sparsely pre-segmented slices, taking into account the complete underlying image data. In addition, Biomedisa enables deep learning for the fully automated segmentation of series of similar samples. It can be used in combination with segmentation tools such as Amira, ImageJ/Fiji and 3D Slicer. If you are using Biomedisa or the data for your research please cite: Lösel, P.D. et al. [Introducing Biomedisa as an open-source online platform for biomedical image segmentation.](https://www.nature.com/articles/s41467-020-19303-w) *Nat. Commun.* **11**, 5577 (2020).

# Hardware requirements
+ One or more NVIDIA GPUs with compute capability 3.0 or higher or an Intel CPU.
+ 32 GB RAM or more (depending on the size of the image data).

# Software requirements
+ [NVIDIA GPU drivers](https://www.nvidia.com/drivers) for GPU support
+ [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) or [Intel Runtime for OpenCL](https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html)

# Installation (command-line-only)
+ [Ubuntu 20.04 + Smart Interpolation + CUDA + GPU](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2004_interpolation_cuda11.3_gpu_cli.md)
+ [Ubuntu 20.04 + Smart Interpolation + OpenCL + CPU](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2004_interpolation_opencl_cpu_cli.md)
+ [Ubuntu 20.04 + Deep Learning](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2004_deeplearning_cuda11.3_cli.md)
+ [Windows 10 + Smart Interpolation + CUDA + GPU](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_interpolation_cuda_gpu_cli.md)
+ [Windows 10 + Smart Interpolation + OpenCL + GPU](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_interpolation_opencl_gpu_cli.md)
+ [Windows 10 + Smart Interpolation + OpenCL + CPU](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_interpolation_opencl_cpu_cli.md)
+ [Windows 10 + Deep Learning](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_deeplearning_cuda11.3_cli.md)

# Full installation (GUI)
+ [Ubuntu 20.04](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2004_cuda11.3.md)
+ [Ubuntu 22.04](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2204_cuda11.8.md)
+ [Windows 10 (21H2 or higher)](https://github.com/biomedisa/biomedisa/blob/master/README/windows11.md)
+ [Windows 11](https://github.com/biomedisa/biomedisa/blob/master/README/windows11.md)

# Biomedisa interpolation

#### Download examples
Download data from the [gallery](https://biomedisa.org/gallery/) or directly as follows:
```
# Brain tumor (small example, good for testing)
wget --no-check-certificate https://biomedisa.org/download/demo/?id=tumor.tif -O ~/Downloads/tumor.tif
wget --no-check-certificate https://biomedisa.org/download/demo/?id=labels.tumor.tif -O ~/Downloads/labels.tumor.tif

# Trigonopterus
wget --no-check-certificate https://biomedisa.org/download/demo/?id=trigonopterus.tif -O ~/Downloads/trigonopterus.tif
wget --no-check-certificate https://biomedisa.org/download/demo/?id=labels.trigonopterus_smart.am -O ~/Downloads/labels.trigonopterus_smart.am

# Mineralized wasp
wget --no-check-certificate https://biomedisa.org/download/demo/?id=NMB_F2875.tif -O ~/Downloads/NMB_F2875.tif
wget --no-check-certificate https://biomedisa.org/download/demo/?id=labels.NMB_F2875.tif -O ~/Downloads/labels.NMB_F2875.tif
```

#### Run the Biomedisa interpolation
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

`--help` or `-h`: show more information and exit

`--version` or `-v`: Biomedisa version

`--nbrw INT`: number of random walks starting at each pre-segmented pixel (default: 10)

`--sorw INT`: steps of a random walk (default: 4000)

`--acwe`: post-processing with active contour (default: False)

`--acwe-alpha FLOAT`: pushing force of active contour (default: 1.0)

`--acwe-smooth INT`: smoothing of active contour (default: 1)

`--acwe-steps INT`: iterations of active contour (default: 3)

`--no-compression`: disable compression of segmentation results (default: False)

`--allaxis` or `-allx`: if pre-segmentation is not exlusively in xy-plane (default: False)

`--denoise` or `-d`: smooth/denoise image data before processing (default: False)

`--uncertainty` or `-u`: return uncertainty of segmentation result (default: False)

`--create_slices` or `-cs`: create slices of segmentation results (default: False)

`--ignore STR`: ignore specific label(s), e.g. "2,5,6" (default: none)

`--only STR`: segment only specific label(s), e.g. "1,3,5" (default: all)

`--smooth INT` or `-s INT`: number of smoothing iterations for segmentation result (default: 0)

`--clean FLOAT` or `-c FLOAT`: remove outliers, e.g. 0.5 means that objects smaller than 50 percent of the size of the largest object will be removed (default: None)

`--fill FLOAT` or `-f FLOAT`: fill holes, e.g. 0.5 means that all holes smaller than 50 percent of the entire label will be filled (default: None)

`--platform STR` or `-p STR`: one of "cuda", "opencl_NVIDIA_GPU", "opencl_Intel_CPU" (default: None)

#### Memory error
If memory errors (either GPU or host memory) occur, you can start the segmentation as follows:
```
python3 ~/git/biomedisa/demo/split_volume.py 'path_to_image' 'path_to_labels' -np 4 -sz 2 -sy 2 -sx 2
```
Where `-n` is the number of GPUs and each axis (`x`,`y` and `z`) is divided into two overlapping parts. The volume is thus divided into `2*2*2=8` subvolumes. These are segmented separately and then reassembled.

# Biomedisa deep learning

#### Download training data, network, and test image
Download data from the [gallery](https://biomedisa.org/gallery/) or directly as follows:
```
# training data
wget --no-check-certificate https://biomedisa.org/download/demo/?id=training_heart.tar -O ~/Downloads/training_heart.tar
wget --no-check-certificate https://biomedisa.org/download/demo/?id=training_heart_labels.tar -O ~/Downloads/training_heart_labels.tar

# extract data
cd ~/Downloads
tar -xf training_heart.tar
tar -xf training_heart_labels.tar

# trained network
wget --no-check-certificate https://biomedisa.org/download/demo/?id=heart.h5 -O ~/Downloads/heart.h5

# test image
wget --no-check-certificate https://biomedisa.org/download/demo/?id=testing_axial_crop_pat13.nii.gz -O ~/Downloads/testing_axial_crop_pat13.nii.gz
```

#### Automatic segmentation using a trained network
To segment the test image, change to the `demo` directory `cd ~/git/biomedisa/demo/` and run
```
# Ubuntu
python3 biomedisa_deeplearning.py ~/Downloads/testing_axial_crop_pat13.nii.gz ~/Downloads/heart.h5 -p -bs 6

# Windows
python biomedisa_deeplearning.py Downloads\testing_axial_crop_pat13.nii.gz Downloads\heart.h5 -p -bs 6
```
`--predict` or `-p`: use the trained neural network to predict the result of the test image. The result will be saved in `Downloads`.

#### Train a neural network for automatic segmentation
To train a neural network, change to the `demo` directory `cd ~/git/biomedisa/demo/` and run
```
# Ubuntu
python3 biomedisa_deeplearning.py ~/Downloads/training_heart ~/Downloads/training_heart_labels -t

# Windows
python biomedisa_deeplearning.py Downloads\training_heart Downloads\training_heart_labels -t
```
`--train` or `-t`: train a neural network. The result will be saved in `Downloads`.

`--epochs INT` or `-e INT`: number of epochs trained (default: 100). 

`--batch-size INT` or `-bs INT`: batch size (default: 24). If you have a memory error, try reducing to 6, for example.

#### Validate the network during training
`--val-images PATH` or `-vi PATH`: path to directory with validation images.

`--val-labels PATH` or `-vl PATH`: path to directory with validation labels.

`--validation-split FLOAT` or `-vs FLOAT`: for example, split your data into 80% training data and 20% validation data with `-vs 0.8`. 

`--early-stopping INT` or `-es INT`: early stopping if there is no improvement after specified number of epochs.

#### Accuracy Assessment: Dice Score vs. Standard Accuracy in Biomedisa
`--val-tf` or `-vt`: use standard pixelwise accuracy provided by TensorFlow. When evaluating accuracy, Biomedisa relies on the Dice score rather than the standard accuracy. The Dice score offers a more reliable assessment by measuring the overlap between the segmented regions, whereas the standard accuracy also considers background classification, which can lead to misleading results, especially when dealing with small segments within a much larger volume. Even if half of the segment is mislabeled, the standard accuracy may still yield a remarkably high value. However, if you still prefer to use the standard accuracy, you can enable it by using this option.

#### Automatic cropping
`--crop-data` or `-cd`: Both the training and inference data should be cropped to the region of interest for best performance. As an alternative to manual cropping, you can use Biomedisa's AI-based automatic cropping. After training, auto cropping is automatically applied to your inference data.

#### Get help
`--help` or `-h`: For more information.

# Biomedisa features

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
xres, yres, zres = get_voxel_spacing(header, data, extension)
print(f'Voxel spacing: x_spacing, y_spacing, z_spacing = {xres}, {yres}, {zres}')

# save stl file
path_to_data = path_to_data.replace(os.path.splitext(path_to_data)[1],'.stl')
save_mesh(path_to_data, data, xres, yres, zres)
```
or call directly
```
python3 git/biomedisa/biomedisa_features/create_mesh.py <path_to_data>
```

#### Other functions
```
import os, sys
sys.path.append(path_to_biomedisa)  # e.g. '/home/<user>/git/biomedisa'
from biomedisa_features.biomedisa_helper import img_resize, clean, fill

# resize image data
zsh, ysh, xsh = data.shape
new_zsh, new_ysh, new_xsh = zsh//2, ysh//2, xsh//2
data = img_resize(data, new_zsh, new_ysh, new_xsh)

# resize label data
label_data = img_resize(label_data, new_zsh, new_ysh, new_xsh, labels=True)

# delete outliers smaller than 90% of the segment
label_data = clean(label_data, 0.9)

# fill holes
label_data = fill(label_data, 0.9)
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

