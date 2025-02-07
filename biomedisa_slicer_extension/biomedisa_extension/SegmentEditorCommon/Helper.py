import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np
from vtkmodules.util.numpy_support import vtk_to_numpy
import subprocess
import sys
import os

class Helper():

    #source: https://discourse.vtk.org/t/convert-vtk-array-to-numpy-array/3152/3
    @staticmethod
    def vtkToNumpy(data):
        temp = vtk_to_numpy(data.GetPointData().GetScalars())
        dims = data.GetDimensions()
        component = data.GetNumberOfScalarComponents()
        if component == 1:
            numpy_data = temp.reshape(dims[2], dims[1], dims[0])
            numpy_data = numpy_data.transpose(0, 1, 2) # Not like in source
        elif component == 3 or component == 4:
            if dims[2] == 1: # a 2D RGB image
                numpy_data = temp.reshape(dims[1], dims[0], component)
                numpy_data = numpy_data.transpose(0, 1, 2)
                numpy_data = np.flipud(numpy_data)
            else:
                raise RuntimeError('unknow type')
        return numpy_data

    @staticmethod
    def expandLabelToMatchInputImage(labelImageData, inputDimensions) -> vtk.vtkImageData:
        # Get extent of the original label image data
        labelExtent = labelImageData.GetExtent()

        # Convert the label image data to a NumPy array
        labelPointData = labelImageData.GetPointData()
        labelVtkArray = labelPointData.GetScalars()
        labelNumpyArray = vtk_np.vtk_to_numpy(labelVtkArray)
        labelNumpyArray = labelNumpyArray.reshape(labelImageData.GetDimensions()[::-1])

        # Initialize the NumPy array for the new label image data with zeros
        newLabelNumpyArray = np.zeros(inputDimensions, dtype=np.uint8)
        newLabelNumpyArray = newLabelNumpyArray.reshape(inputDimensions[::-1])

        # Copy label data to the new image data at the correct position
        zmin, zmax = labelExtent[4], labelExtent[5] + 1
        ymin, ymax = labelExtent[2], labelExtent[3] + 1
        xmin, xmax = labelExtent[0], labelExtent[1] + 1
        newLabelNumpyArray[zmin:zmax, ymin:ymax, xmin:xmax] = labelNumpyArray

        return newLabelNumpyArray

    @staticmethod
    def crop(image: np.ndarray,
             x_min: int, x_max: int,
             y_min: int, y_max: int,
             z_min: int, z_max: int) -> np.ndarray:
        image = image[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1].copy()
        return image

    @staticmethod
    def embed(image: np.ndarray,
             dimensions,
             x_min: int, x_max: int,
             y_min: int, y_max: int,
             z_min: int, z_max: int) -> np.ndarray:
        fullimage = np.zeros((dimensions[2], dimensions[1], dimensions[0]), dtype=image.dtype)
        fullimage[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1] = image
        return fullimage

    @staticmethod
    def get_python_version(env_path):
        try:
            # Run the Python executable of a pyhthon environment with --version
            result = subprocess.run(
                [env_path, '-c', "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def module_exists(module_name):
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    @staticmethod
    def check_wsl_path_exists(path, wsl_path):
        try:
            # Run the WSL command with `bash -c` to check the path
            result = subprocess.run(
                wsl_path + [f"test -e {path} && echo 'Exists' || echo 'Does not exist'"],
                capture_output=True,
                text=True,
                check=True
            )
            # Check the output
            output = result.stdout.strip()
            return output == "Exists"
        except subprocess.CalledProcessError as e:
            print(f"Error checking path: {e}")
            return False

    @staticmethod
    def build_environment(cmd):
        # Import config
        try:
            from biomedisa_extension.config import python_path, lib_path, wsl_path
        except:
            from biomedisa_extension.config_template import python_path, lib_path, wsl_path

        # Create a clean environment
        new_env = os.environ.copy()

        # Windows using WSL
        if os.name == "nt" and wsl_path!=False:
            if wsl_path==None:
                wsl_path = ['wsl','-e','bash','-c']
            if python_path==None:
                if os.path.exists(os.path.expanduser("~")+"/biomedisa_env/bin"):
                    python_path = (os.path.expanduser("~")+"/biomedisa_env/bin/python").replace('\\','/').replace('C:','/mnt/c')
                elif Helper.check_wsl_path_exists("~/biomedisa_env/bin", wsl_path):
                    python_path = "~/biomedisa_env/bin/python"
                else:
                    python_path = "/usr/bin/python3"
            if lib_path==None:
                lib_path = "export CUDA_HOME=/usr/local/cuda && export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 && export PATH=${CUDA_HOME}/bin:${PATH}"
                if python_path == "/usr/bin/python3":
                    lib_path = lib_path + " && export PATH=${HOME}/.local/bin:${PATH}"
            cmd = wsl_path + [lib_path + " && " + python_path + " " + (" ").join(cmd)]
            print("Command used:", cmd)

        # Linux or Windows without WSL
        else:
            # Path to the desired Python executable
            if python_path==None:
                if os.path.exists(os.path.expanduser("~")+"/biomedisa_env/bin/python"):
                    python_path = os.path.expanduser("~")+"/biomedisa_env/bin/python"
                    python_version = Helper.get_python_version(python_path)
                    lib_path = os.path.expanduser("~")+f"/biomedisa_env/lib/python{python_version}/site-package"
                else:
                    python_path = "/usr/bin/python3"
                    python_version = Helper.get_python_version(python_path)
                    lib_path = os.path.expanduser("~")+f"/.local/lib/python{python_version}/site-packages"

            # Remove environment variables that may interfere
            for var in ["PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH"]:
                new_env.pop(var, None)

            # Set new pythonpath
            new_env["PYTHONPATH"] = lib_path

            # Run the Python 3 subprocess
            subprocess.Popen(
                [python_path, "-c", "import sys; print(sys.version); print(sys.executable); print(sys.path)"],
                env=new_env
            )

            # command
            cmd = [python_path] + cmd
        return cmd, new_env

