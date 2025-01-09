import os

# default (automatic detection)
env_path = None
lib_path = None

# Windows using WSL
#env_path = "wsl -d Ubuntu-22.04 -e bash -c 'export CUDA_HOME=/usr/local/cuda-12.6 && export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 && export PATH=${CUDA_HOME}/bin:${PATH} && ~/biomedisa_env/bin/python'"

# Ubuntu
# 1st example
#env_path = os.path.expanduser("~")+"/biomedisa_env/bin/python"
#lib_path = os.path.expanduser("~")+"/biomedisa_env/lib/python3.10/site-package"

# 2nd example
#env_path = "/usr/bin/python3"
#lib_path = os.path.expanduser("~")+"/.local/lib/python3.10/site-packages"

