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
+ At least one [NVIDIA](https://www.nvidia.com/) Graphics Procissing Unit (GPU) with compute capability 3.0 or higher.
+ 32 GB RAM or more (strongly depends on the size of the processed images).

# Software requirements
+ [NVIDIA GPU drivers](https://www.nvidia.com/drivers) - 455.x or higher.
+ [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) - CUDA 11.0 or higher.

# Installation (command-line-only)
+ [Ubuntu 18.04.5 + CUDA 11.0 (Pascal, Volta)](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu18.04+cuda11.0_cli.md)
+ [Ubuntu 20.04.3 + CUDA 11.0 (Pascal, Volta)](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu20.04+cuda11.0_cli.md)
+ [Ubuntu 20.04.3 + CUDA 11.4 (Ampere)](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu20.04+cuda11.4_cli.md)
+ [Windows 10 + CUDA 11.0 (Pascal, Volta)](https://github.com/biomedisa/biomedisa/blob/master/README/windows10+cuda11.0_cli.md)

# Full installation (GUI)
+ [Ubuntu 18.04.5 + CUDA 11.0 (Pascal, Volta)](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu1804+cuda11.0.md)
+ [Ubuntu 20.04.3 + CUDA 11.4 (Ampere)](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2004+cuda11.4.md)
+ [Windows 10 (21H1 or lower) + CUDA 11.0 (Pascal, Volta)](https://github.com/biomedisa/biomedisa/blob/master/README/windows10+cuda11.0.md)
+ [Windows 10 (21H2)](https://github.com/biomedisa/biomedisa/blob/master/README/windows11.md)
+ [Windows 11](https://github.com/biomedisa/biomedisa/blob/master/README/windows11.md)

# Run interpolation examples

#### Small example
Change to the demo directory.
```
cd git/biomedisa/demo
```

Run a simple example (~4 seconds). The result will be saved as `final.tumor.tif`.
```
# Ubuntu
python3 biomedisa_interpolation.py tumor.tif labels.tumor.tif

# Windows
python biomedisa_interpolation.py tumor.tif labels.tumor.tif
```

#### Further examples
Download the examples from the [gallery](https://biomedisa.org/gallery/) or directly as follows:
```
# Trigonopterus
wget --no-check-certificate https://biomedisa.org/download/demo/?id=trigonopterus.tif -O trigonopterus.tif
wget --no-check-certificate https://biomedisa.org/download/demo/?id=labels.trigonopterus_smart.am -O labels.trigonopterus_smart.am

# Wasp from amber
wget --no-check-certificate https://biomedisa.org/download/demo/?id=wasp_from_amber.tif -O wasp_from_amber.tif
wget --no-check-certificate https://biomedisa.org/download/demo/?id=labels.wasp_from_amber.am -O labels.wasp_from_amber.am

# Cockroach
wget --no-check-certificate https://biomedisa.org/download/demo/?id=cockroach.tif -O cockroach.tif
wget --no-check-certificate https://biomedisa.org/download/demo/?id=labels.cockroach.am -O labels.cockroach.am

# Theropod claw
wget --no-check-certificate https://biomedisa.org/download/demo/?id=theropod_claw.tif -O theropod_claw.tif
wget --no-check-certificate https://biomedisa.org/download/demo/?id=labels.theropod_claw.tif -O labels.theropod_claw.tif

# Mineralized wasp
wget --no-check-certificate https://biomedisa.org/download/demo/?id=NMB_F2875.tif -O NMB_F2875.tif
wget --no-check-certificate https://biomedisa.org/download/demo/?id=labels.NMB_F2875.tif -O labels.NMB_F2875.tif

# Bull ant
wget --no-check-certificate https://biomedisa.org/download/demo/?id=bull_ant_queen.am -O bull_ant.tif
wget --no-check-certificate https://biomedisa.org/download/demo/?id=labels.bull_ant_queen_head.am -O labels.bull_ant_head.am
```

Run the segmentation using e.g. 4 GPUs.
```
mpiexec -n 4 python3 biomedisa_interpolation.py NMB_F2875.tif labels.NMB_F2875.tif
```

Obtain uncertainty and smoothing as optional results.
```
mpiexec -n 4 python3 biomedisa_interpolation.py NMB_F2875.tif labels.NMB_F2875.tif -uq -s 100
```

Use pre-segmentation with different orientations (not exclusively xy-plane).
```
mpiexec -n 4 python3 'path_to_image' 'path_to_labels' -allx
```

# Run AI example
Change to the demo directory.
```
cd git/biomedisa/demo
```

Download the Deep Learning example `human heart` from the [gallery](https://biomedisa.org/gallery/) or directly as follows:
```
wget --no-check-certificate https://biomedisa.org/download/demo/?id=training_heart.tar -O training_heart.tar
wget --no-check-certificate https://biomedisa.org/download/demo/?id=training_heart_labels.tar -O training_heart_labels.tar
wget --no-check-certificate https://biomedisa.org/download/demo/?id=testing_axial_crop_pat13.nii.gz -O testing_axial_crop_pat13.nii.gz
```

Extract the data. This creates a `heart` directory containing the image data and a `label` directory containing the label data.
```
tar -xf training_heart.tar
tar -xf training_heart_labels.tar
```

Train a neural network with 200 epochs and batch size (-bs) of 24. The result will be saved as `heart.h5`. If you have only a single GPU, reduce batch size to 6.
```
# Ubuntu
python3 biomedisa_deeplearning.py heart label -train -epochs 200 -bs 24

# Windows
python biomedisa_deeplearning.py heart label -train -epochs 200 -bs 24
```

Alternatively, you can download the trained network from the [gallery](https://biomedisa.org/gallery/) or directly with the command
```
wget --no-check-certificate https://biomedisa.org/download/demo/?id=heart.h5 -O heart.h5
```

Use the trained network to predict the result of the test image. The result will be saved as `final.testing_axial_crop_pat13.tif`.
```
# Ubuntu
python3 biomedisa_deeplearning.py testing_axial_crop_pat13.nii.gz heart.h5 -predict -bs 6

# Windows
python biomedisa_deeplearning.py testing_axial_crop_pat13.nii.gz heart.h5 -predict -bs 6
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

