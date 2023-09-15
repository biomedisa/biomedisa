# Ubuntu 22.04 LTS + Deep Learning + CUDA 11.8 (command-line-only)

- [Install Biomedisa](#install-biomedisa)
- [Install TensorFlow](#install-tensorflow)
- [Biomedisa example](#biomedisa-example)

#### Install Biomedisa
Follow Biomedisa [installation](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2204_interpolation_cuda11.8_gpu_cli.md).

#### Install TensorFlow
```
# Install development and runtime libraries.
sudo apt-get install --no-install-recommends \
    libcudnn8=8.8.0.121-1+cuda11.8 \
    libcudnn8-dev=8.8.0.121-1+cuda11.8

# Install TensorRT. Requires that libcudnn8 is installed above.
sudo apt-get install --no-install-recommends libnvinfer8=8.5.3-1+cuda11.8 \
    libnvinfer-dev=8.5.3-1+cuda11.8 \
    libnvinfer-plugin8=8.5.3-1+cuda11.8

# OPTIONAL: hold packages to avoid crash after system update
sudo apt-mark hold libcudnn8 libcudnn8-dev libnvinfer-dev libnvinfer-plugin8 libnvinfer8 cuda-11-8

# Install TensorFlow
sudo -H pip3 install tensorflow==2.13.0
```

#### Biomedisa example
Download test files from [Gallery](https://biomedisa.de/gallery/).
```
python3 git/biomedisa/demo/biomedisa_deeplearning.py Downloads/testing_axial_crop_pat13.nii.gz Downloads/heart.h5 -p
```

