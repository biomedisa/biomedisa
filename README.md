# Biomedisa

- [Overview](#overview)
- [Hardware requirements](#hardware-requirements)
- [Software requirements](#software-requirements)
- [Installation (command-line-only)](#installation-command-line-only)
- [Full installation (GUI)](#full-installation-gui)
- [Run interpolation examples](#run-interpolation-examples)
- [Run AI example](#run-ai-example)
- [Update Biomedisa](#update-biomedisa)
- [Releases](#releases)
- [Authors](#authors)
- [FAQ](#faq)
- [Citation](#citation)
- [License](#license)

# Overview
Biomedisa (https://biomedisa.org) is a free and easy-to-use open-source online platform for segmenting large volumetric images, e.g. CT and MRI scans, at Heidelberg University and the Heidelberg Institute for Theoretical Studies (HITS). The segmentation is based on a smart interpolation of sparsely pre-segmented slices taking into account the complete underlying image data. It can be used in addition to segmentation tools like Amira, ImageJ/Fiji and MITK. Biomedisa finds its root in the projects ASTOR and NOVA funded by the Federal Ministry of Education and Research (BMBF). If you are using Biomedisa for your research please cite: Lösel, P.D. et al. [Introducing Biomedisa as an open-source online platform for biomedical image segmentation.](https://www.nature.com/articles/s41467-020-19303-w) *Nat. Commun.* **11**, 5577 (2020).

# Hardware requirements
+ At least one [NVIDIA](https://www.nvidia.com/) Graphics Processing Unit (GPU) with compute capability 3.0 or higher.
+ 32 GB RAM or more (strongly depends on the size of the processed images).

# Software requirements
+ [NVIDIA GPU drivers](https://www.nvidia.com/drivers) - 455.x or higher.
+ [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) - CUDA 11.0 or higher.

# Installation (command-line-only)
+ [Ubuntu 18.04.5 + CUDA 11.0 (Pascal, Volta)](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu1804+cuda11.0_cli.md)
+ [Ubuntu 20.04.3 + CUDA 11.0 (Pascal, Volta)](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2004+cuda11.0_cli.md)
+ [Ubuntu 20.04.3 + CUDA 11.4 (Ampere)](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2004+cuda11.4_cli.md)
+ [Windows 10 + CUDA 11.0 (Pascal, Volta)](https://github.com/biomedisa/biomedisa/blob/master/README/windows10+cuda11.0_cli.md)

# Full installation (GUI)
+ [Ubuntu 18.04.5 + CUDA 11.0 (Pascal, Volta)](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu1804+cuda11.0.md)
+ [Ubuntu 20.04.3 + CUDA 11.4 (Ampere)](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2004+cuda11.4.md)
+ [Windows 10 (21H1 or lower) + CUDA 11.0 (Pascal, Volta)](https://github.com/biomedisa/biomedisa/blob/master/README/windows10+cuda11.0.md)
+ [Windows 10 (21H2)](https://github.com/biomedisa/biomedisa/blob/master/README/windows11.md)
+ [Windows 11](https://github.com/biomedisa/biomedisa/blob/master/README/windows11.md)

# Run interpolation examples

#### Small example
Download the tumor test example from the [gallery](https://biomedisa.org/gallery/) or directly as follows:
```
# Brain tumor
wget --no-check-certificate https://biomedisa.org/download/demo/?id=tumor.tif -O ~/Downloads/tumor.tif
wget --no-check-certificate https://biomedisa.org/download/demo/?id=labels.tumor.tif -O ~/Downloads/labels.tumor.tif
```

Run Biomedisa (~3 seconds). The result will be saved in `Downloads` as `final.tumor.tif`.
```
# Ubuntu
python3 ~/git/biomedisa/demo/biomedisa_interpolation.py ~/Downloads/tumor.tif ~/Downloads/labels.tumor.tif

# Windows
python git/biomedisa/demo/biomedisa_interpolation.py Downloads/tumor.tif Downloads/labels.tumor.tif

# Windows 10 21H2 and 11 (replace "Biomedisa-2x.xx.x" with your Biomedisa version, use "wsl -l -v" to get the version)
cd AppData\Biomedisa-2x.xx.x
biomedisa_interpolation.cmd Downloads/tumor.tif Downloads/labels.tumor.tif
```

#### Further examples
Download further examples from the [gallery](https://biomedisa.org/gallery/) or directly as follows:
```
# Trigonopterus
wget --no-check-certificate https://biomedisa.org/download/demo/?id=trigonopterus.tif -O ~/Downloads/trigonopterus.tif
wget --no-check-certificate https://biomedisa.org/download/demo/?id=labels.trigonopterus_smart.am -O ~/Downloads/labels.trigonopterus_smart.am

# Mineralized wasp
wget --no-check-certificate https://biomedisa.org/download/demo/?id=NMB_F2875.tif -O ~/Downloads/NMB_F2875.tif
wget --no-check-certificate https://biomedisa.org/download/demo/?id=labels.NMB_F2875.tif -O ~/Downloads/labels.NMB_F2875.tif
```

Run the segmentation using e.g. 4 GPUs.
```
# Ubuntu
mpiexec -n 4 python3 ~/git/biomedisa/demo/biomedisa_interpolation.py ~/Downloads/NMB_F2875.tif ~/Downloads/labels.NMB_F2875.tif

# Windows 10 21H2 and 11 (replace "Biomedisa-2x.xx.x" with your Biomedisa version, use "wsl -l -v" to get the version)
cd AppData\Biomedisa-2x.xx.x
biomedisa_interpolation.cmd -n 4 Downloads/NMB_F2875.tif Downloads/labels.NMB_F2875.tif
```

Obtain uncertainty and smoothing as optional results.
```
mpiexec -n 4 python3 ~/git/biomedisa/demo/biomedisa_interpolation.py ~/Downloads/NMB_F2875.tif ~/Downloads/labels.NMB_F2875.tif -uq -s 100
```

Use pre-segmentation with different orientations (not exclusively xy-plane).
```
mpiexec -n 4 python3 ~/git/biomedisa/demo/biomedisa_interpolation.py 'path_to_image' 'path_to_labels' -allx
```

# Run AI example

Download a trained neural network and a test image from the [gallery](https://biomedisa.org/gallery/) or directly as follows:
```
wget --no-check-certificate https://biomedisa.org/download/demo/?id=heart.h5 -O ~/Downloads/heart.h5
wget --no-check-certificate https://biomedisa.org/download/demo/?id=testing_axial_crop_pat13.nii.gz -O ~/Downloads/testing_axial_crop_pat13.nii.gz
```

Use the trained neural network to **predict the result** of the test image. The result will be saved in `Downloads` as `final.testing_axial_crop_pat13.tif`.
```
# Ubuntu
python3 ~/git/biomedisa/demo/biomedisa_deeplearning.py ~/Downloads/testing_axial_crop_pat13.nii.gz ~/Downloads/heart.h5 -predict -bs 6

# Windows
python git/biomedisa/demo/biomedisa_deeplearning.py Downloads/testing_axial_crop_pat13.nii.gz Downloads/heart.h5 -predict -bs 6

# Windows 10 21H2 and 11 (replace "Biomedisa-2x.xx.x" with your Biomedisa version, use "wsl -l -v" to get the version)
wsl -d Biomedisa-2x.xx.x -u biomedisa python3 ~/git/biomedisa/demo/biomedisa_deeplearning.py Downloads/testing_axial_crop_pat13.nii.gz Downloads/heart.h5 -p -bs 6
```

To train the neural network yourself, download and extract the training data from the [gallery](https://biomedisa.org/gallery/) or directly as follows:
```
wget --no-check-certificate https://biomedisa.org/download/demo/?id=training_heart.tar -O ~/Downloads/training_heart.tar
wget --no-check-certificate https://biomedisa.org/download/demo/?id=training_heart_labels.tar -O ~/Downloads/training_heart_labels.tar
cd ~/Downloads
tar -xf training_heart.tar
tar -xf training_heart_labels.tar
```

**Train a neural network** with 200 epochs and batch size (-bs) of 24. The result will be saved in `Downloads` as `heart.h5`. If you have a single GPU or low memory, reduce the batch size to 6.
```
# Ubuntu
python3 ~/git/biomedisa/demo/biomedisa_deeplearning.py ~/Downloads/training_heart ~/Downloads/training_heart_labels -train -epochs 200 -bs 24

# Windows
python git/biomedisa/demo/biomedisa_deeplearning.py Downloads/training_heart Downloads/training_heart_labels -train -epochs 200 -bs 24

# Windows 10 21H2 and 11 (replace "Biomedisa-2x.xx.x" with your Biomedisa version, use "wsl -l -v" to get the version)
wsl -d Biomedisa-2x.xx.x -u biomedisa python3 ~/git/biomedisa/demo/biomedisa_deeplearning.py Downloads/training_heart Downloads/training_heart_labels -train -epochs 200 -bs 24
```

# Update Biomedisa
If you have used `git clone`, change to the Biomedisa directory and make a pull request.
```
cd git/biomedisa
git pull
```

If you have fully installed Biomedisa (including MySQL database), update the database.
```
python3 manage.py migrate
```

If you have installed an [Apache Server](https://github.com/biomedisa/biomedisa/blob/master/README/APACHE_SERVER.md), restart the server.
```
sudo service apache2 restart
```

# Releases

For the versions available, see the [tags on this repository](https://github.com/biomedisa/biomedisa/tags). 

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

