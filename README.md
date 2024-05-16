[![biomedisa](https://raw.githubusercontent.com/biomedisa/biomedisa/master/biomedisa_app/static/biomedisa_logo.svg)](https://biomedisa.info)
-----------
- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Installation (command-line based)](#installation-command-line-based)
- [Installation (browser based)](#installation-browser-based)
- [Download Data](#download-data)
- [Revisions](#revisions)
- [Quickstart](#quickstart)
- [Smart Interpolation](#smart-interpolation)
- [Deep Learning](#deep-learning)
- [Biomedisa Features](#biomedisa-features)
- [Authors](#authors)
- [FAQ](#faq)
- [Citation](#citation)
- [License](#license)

## Overview
Biomedisa (https://biomedisa.info) is a free and easy-to-use open-source application for segmenting large volumetric images, e.g. CT and MRI scans, developed at [The Australian National University CTLab](https://ctlab.anu.edu.au/). Biomedisa's semi-automated segmentation is based on a smart interpolation of sparsely pre-segmented slices, taking into account the complete underlying image data. In addition, Biomedisa enables deep learning for the fully automated segmentation of series of similar samples. It can be used in combination with segmentation tools such as Amira/Avizo, ImageJ/Fiji and 3D Slicer. If you are using Biomedisa or the data for your research please cite: Lösel, P.D. et al. [Introducing Biomedisa as an open-source online platform for biomedical image segmentation.](https://www.nature.com/articles/s41467-020-19303-w) *Nat. Commun.* **11**, 5577 (2020).

## Hardware Requirements
+ One or more NVIDIA GPUs with compute capability 3.0 or higher or an Intel CPU.

## Installation (command-line based)
+ [Ubuntu 22.04 + CUDA + GPU (recommended)](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2204_cuda11.8_gpu_cli.md)
+ [Ubuntu 22.04 + OpenCL + CPU (smart interpolation only and very slow)](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2204_opencl_cpu_cli.md)
+ [Windows 10 + CUDA + GPU (recommended)](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_cuda_gpu_cli.md)
+ [Windows 10 + OpenCL + GPU (easy to install but lacks features like allaxis, smoothing, uncertainty, optimized GPU memory usage)](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_opencl_gpu_cli.md)
+ [Windows 10 + OpenCL + CPU (very slow)](https://github.com/biomedisa/biomedisa/blob/master/README/windows10_opencl_cpu_cli.md)

## Installation (browser based)
+ [Ubuntu 22.04](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2204_cuda11.8.md)

## Download Data
+ Download the data from our [gallery](https://biomedisa.info/gallery/)

## Revisions
2024.05.22
+ Pip is the preferred installation method
+ Commands, module names and imports have been changed to conform to the Pip standard
+ For versions <=2023.09.1 please check [README](https://github.com/biomedisa/biomedisa/blob/master/README/deprecated/README_2023.09.1.md)

## Quickstart
Install the Biomedisa package from the [Python Package Index](https://pypi.org/project/biomedisa/):
```
python -m pip install -U biomedisa
```
For smart interpolation and deep Learning modules, follow the installation instructions above.

## Smart Interpolation
+ [Parameters and Examples](https://github.com/biomedisa/biomedisa/blob/master/README/smart_interpolation.md)

#### Python example
```python
from biomedisa.features.biomedisa_helper import load_data, save_data
from biomedisa.interpolation import smart_interpolation

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
python -m biomedisa.interpolation C:\Users\%USERNAME%\Downloads\tumor.tif C:\Users\%USERNAME%\Downloads\labels.tumor.tif
```
If pre-segmentation is not exclusively in the XY plane:
```
python -m biomedisa.interpolation C:\Users\%USERNAME%\Downloads\tumor.tif C:\Users\%USERNAME%\Downloads\labels.tumor.tif --allaxis
```

## Deep Learning
+ [Parameters and Examples](https://github.com/biomedisa/biomedisa/blob/master/README/deep_learning.md)

#### Python example (training)
```python
from biomedisa.features.biomedisa_helper import load_data
from biomedisa.deeplearning import deep_learning

# load image data
img1, _ = load_data('Head1.am')
img2, _ = load_data('Head2.am')
img_data = [img1, img2]

# load label data and header information to be stored in the network file (optional)
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
Start training with a batch size of 12:
```
python -m biomedisa.deeplearning C:\Users\%USERNAME%\Downloads\training_heart C:\Users\%USERNAME%\Downloads\training_heart_labels -t -bs=12
```
Monitor training progress using validation data:
```
python -m biomedisa.deeplearning C:\Users\%USERNAME%\Downloads\training_heart C:\Users\%USERNAME%\Downloads\training_heart_labels -t -vi=C:\Users\%USERNAME%\Downloads\val_img -vl=C:\Users\%USERNAME%\Downloads\val_labels
```
If running into ResourceExhaustedError due to out of memory (OOM), try to use a smaller batch size.

#### Python example (prediction)
```python
from biomedisa.features.biomedisa_helper import load_data, save_data
from biomedisa.deeplearning import deep_learning

# load data
img, _ = load_data('Head5.am')

# deep learning
results = deep_learning(img, predict=True,
        path_to_model='honeybees.h5', batch_size=6)

# save result
save_data('final.Head5.am', results['regular'], results['header'])
```

#### Command-line based (prediction)
```
python -m biomedisa.deeplearning C:\Users\%USERNAME%\Downloads\testing_axial_crop_pat13.nii.gz C:\Users\%USERNAME%\Downloads\heart.h5 -p
```

## Biomedisa Features

#### Load and save data (such as Amira Mesh, TIFF, NRRD, NIfTI or DICOM)
```python
from biomedisa.features.biomedisa_helper import load_data, save_data

# load data as numpy array
# for DICOM, PNG files, or similar formats, 'path_to_data' must reference
# either a directory or a ZIP file containing the image slices
data, header = load_data(path_to_data)

# save data (for TIFF, header=None)
save_data(path_to_data, data, header)
```

#### Create STL mesh from segmentation (label values are saved as attributes)
```python
from biomedisa.features.biomedisa_helper import load_data, save_data
from biomedisa.mesh import get_voxel_spacing, save_mesh

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
python -m biomedisa.mesh <path_to_data>
```

#### Options
`--poly_reduction` or `-pr`: Reduce number of polygons by this factor (default: 0.9)

`--smoothing_iterations` or `-s`: Iteration steps for smoothing (default: 15)

`--x_res` or `-xres`: Voxel spacing/resolution x-axis (default: None)

`--y_res` or `-yres`: Voxel spacing/resolution y-axis (default: None)

`--z_res` or `-zres`: Voxel spacing/resolution z-axis (default: None)

#### Resize data
```python
from biomedisa.features.biomedisa_helper import img_resize

# resize image data
zsh, ysh, xsh = data.shape
new_zsh, new_ysh, new_xsh = zsh//2, ysh//2, xsh//2
data = img_resize(data, new_zsh, new_ysh, new_xsh)

# resize label data
label_data = img_resize(label_data, new_zsh, new_ysh, new_xsh, labels=True)
```

#### Remove outliers and fill holes
```python
from biomedisa.features.biomedisa_helper import clean, fill

# delete outliers smaller than 90% of the segment
label_data = clean(label_data, 0.9)

# fill holes
label_data = fill(label_data, 0.9)
```

#### Accuracy assessment
```python
from biomedisa.features.biomedisa_helper import Dice_score, ASSD
dice = Dice_score(ground_truth, result)
assd = ASSD(ground_truth, result)
```

## Authors

* **Philipp D. Lösel**

See also the list of [contributors](https://github.com/biomedisa/biomedisa/blob/master/credits.md) who participated in this project.

## FAQ
Frequently asked questions can be found at: https://biomedisa.info/faq/.

## Citation

If you use Biomedisa or the data, please cite the following paper:

`Lösel, P.D. et al. Introducing Biomedisa as an open-source online platform for biomedical image segmentation. Nat. Commun. 11, 5577 (2020).` https://doi.org/10.1038/s41467-020-19303-w

If you use Biomedisa's Deep Learning, you may also cite:

`Lösel, P.D. et al. Natural variability in bee brain size and symmetry revealed by micro-CT imaging and deep learning. PLoS Comput. Biol. 19, e1011529 (2023).` https://doi.org/10.1371/journal.pcbi.1011529

If you use Biomedisa's Smart Interpolation, you can also cite the initial description of this method:

`Lösel, P. & Heuveline, V. Enhancing a diffusion algorithm for 4D image segmentation using local information. Proc. SPIE 9784, 97842L (2016).` https://doi.org/10.1117/12.2216202

## License

This project is covered under the **EUROPEAN UNION PUBLIC LICENCE v. 1.2 (EUPL)**.

