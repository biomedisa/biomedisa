# Ubuntu 20.04 LTS + Deep Learning + CUDA 11.3 (command-line-only)

- [Install Biomedisa](#install-biomedisa)
- [Install TensorFlow](#install-tensorflow)
- [Biomedisa example](#biomedisa-example)

#### Install Biomedisa
Follow Biomedisa [installation](https://github.com/biomedisa/biomedisa/blob/master/README/ubuntu2004_interpolation_cuda11.3_cli.md).

#### Install TensorFlow
```
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu2004_1.0.0-1_amd64.deb
sudo apt-get update

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libnvinfer8_8.0.3-1+cuda11.3_amd64.deb
sudo apt install ./libnvinfer8_8.0.3-1+cuda11.3_amd64.deb
sudo apt-get update

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    libcudnn8=8.2.1.32-1+cuda11.3  \
    libcudnn8-dev=8.2.1.32-1+cuda11.3

# Install TensorRT. Requires that libcudnn8 is installed above.
sudo apt-get install -y --no-install-recommends libnvinfer8=8.0.3-1+cuda11.3 \
    libnvinfer-dev=8.0.3-1+cuda11.3 \
    libnvinfer-plugin8=8.0.3-1+cuda11.3

# Install TensorFlow
sudo -H pip3 install --upgrade tensorflow-gpu
```

#### Biomedisa example
Download test files from [Gallery](https://biomedisa.de/gallery/).
```
python3 git/biomedisa/demo/biomedisa_deeplearning.py Downloads/testing_axial_crop_pat13.nii.gz Downloads/heart.h5 -p
```

