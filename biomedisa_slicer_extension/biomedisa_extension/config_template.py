import getpass
username = getpass.getuser()
#-------------------------------
# default (automatic detection)
#-------------------------------
python_path = None
lib_path = None
wsl_path = None

#-------------------------------
# Windows using WSL
#-------------------------------
'''Virtual Python Environment'''
#python_path = "/home/$USER/biomedisa_env/bin/python" # if the environment is in the WSL home directory
#python_path = f"/mnt/c/Users/{username}/biomedisa_env/bin/python" # if the environment is in the Windows User directory
#lib_path = "export CUDA_HOME=/usr/local/cuda-12.6 && export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 && export PATH=${CUDA_HOME}/bin:${PATH}"
#wsl_path = ["wsl","-d","Ubuntu-22.04","-e","bash","-c"]

'''System Python'''
#python_path = "/usr/bin/python3"
#lib_path = "export CUDA_HOME=/usr/local/cuda-12.6 && export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 && export PATH=${CUDA_HOME}/bin:${PATH} && export PATH=${HOME}/.local/bin:${PATH}"
#wsl_path = ["wsl","-d","Ubuntu-22.04","-e","bash","-c"]

#-------------------------------
# Windows without WSL (not recommended)
#-------------------------------
'''Conda Environment'''
#python_path = f"C:/Users/{username}/anaconda3/envs/biomedisa/python.exe"
#lib_path = f"C:/Users/{username}/anaconda3/envs/biomedisa/lib/site-packages"
#wsl_path = False

'''Slicer environment'''
#python_path = f"C:/Users/{username}/AppData/Local/slicer.org/Slicer 5.6.2/bin/PythonSlicer.exe"
#lib_path = f"C:/Users/{username}/AppData/Local/slicer.org/Slicer 5.6.2/lib/Python/Lib/site-packages"
#wsl_path = False

#-------------------------------
# Linux
#-------------------------------
'''Virtual Python Environment'''
#python_path = f"/home/{username}/biomedisa_env/bin/python"
#lib_path = f"/home/{username}/biomedisa_env/lib/python3.10/site-packages"
#wsl_path = None

'''System Python'''
#python_path = "/usr/bin/python3"
#lib_path = f"/home/{username}/.local/lib/python3.10/site-packages"
#wsl_path = None

