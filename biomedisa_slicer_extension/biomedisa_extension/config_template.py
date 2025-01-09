import getpass

# default (automatic detection)
env_path = None
lib_path = None

# Windows using WSL
username = getpass.getuser()
python_path = "/home/$USER/biomedisa_env/bin/python" # if environment in WSL home directory
python_path = f"/mnt/c/Users/{username}/biomedisa_env/bin/python" # if environment in Windows User directory
#env_path = ["wsl","-d","Ubuntu-22.04","-e","bash","-c","export CUDA_HOME=/usr/local/cuda-12.6 && export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 && export PATH=${CUDA_HOME}/bin:${PATH} && " + python_path]

# Ubuntu
# 1st example
#env_path = "/home/$USER/biomedisa_env/bin/python"
#lib_path = "/home/$USER/biomedisa_env/lib/python3.10/site-package"

# 2nd example
#env_path = "/usr/bin/python3"
#lib_path = "/home/$USER/.local/lib/python3.10/site-packages"

