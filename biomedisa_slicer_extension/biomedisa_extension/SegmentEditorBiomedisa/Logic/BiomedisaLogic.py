import platform
import numpy as np
import vtk, slicer
import vtk.util.numpy_support as vtk_np
from vtkmodules.util.numpy_support import vtk_to_numpy
from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLLabelMapVolumeNode
from Logic.BiomedisaParameter import BiomedisaParameter
from SegmentEditorCommon.Helper import Helper
import subprocess
import tempfile
from tifffile import imread, imwrite
import sys
import os
import json

class BiomedisaLogic():

    def _getBinaryLabelMap(label: np.array, direction_matrix: np.array, labels: vtkMRMLLabelMapVolumeNode) -> vtkMRMLLabelMapVolumeNode:
        vtkImageData = slicer.vtkOrientedImageData()
        vtkImageData.SetDimensions(label.shape[2], label.shape[1], label.shape[0])
        vtkImageData.SetDirections(direction_matrix)
        vtkImageData.SetOrigin(labels.GetOrigin())
        vtkImageData.SetSpacing(labels.GetSpacing())
        vtkImageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        vtkArray = vtk_np.numpy_to_vtk(num_array=label.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtkImageData.GetPointData().SetScalars(vtkArray)
        return vtkImageData

    def _getBinaryLabelMaps(labelmapArray: np.array, direction_matrix: np.array,
            labels: vtkMRMLLabelMapVolumeNode, uniqueLabels: np.array) -> list:
        labelMapList = []
        for label in uniqueLabels:
            if label == 0:
                continue
            binaryLabelmapArray = np.where(labelmapArray == label, 1, 0).astype(np.uint8)
            vtkBinaryLabelmap  = BiomedisaLogic._getBinaryLabelMap(binaryLabelmapArray, direction_matrix, labels)
            print(f"label: {label}")
            labelMapList.append(vtkBinaryLabelmap)
        return labelMapList

    def getLabeledSlices(input: vtkMRMLScalarVolumeNode, labels: vtkMRMLLabelMapVolumeNode):
        extendedLabel = Helper.expandLabelToMatchInputImage(labels, input.GetDimensions())
        numpyLabels = Helper.vtkToNumpy(extendedLabel)
        from biomedisa.interpolation import read_labeled_slices
        labeledSlices, _ = read_labeled_slices(numpyLabels)
        return labeledSlices

    def unify_to_identity(array, direction_matrix):
        # Determine the permutation of axes
        permutation = np.argmax(np.abs(direction_matrix), axis=0)
        # Determine the flips
        flips = np.sum(direction_matrix, axis=0) < 0
        # Transpose the array according to the permutation
        transposed_array = np.transpose(array, permutation)
        # Flip the required axes
        for axis in np.where(flips)[0]:
            transposed_array = np.flip(transposed_array, axis=axis)
        return transposed_array

    def reverse_unify_to_identity(array, direction_matrix):
        # Determine the permutation of axes used to unify to identity
        permutation = np.argmax(np.abs(direction_matrix), axis=0)
        # Determine the flips used
        flips = np.sum(direction_matrix, axis=0) < 0
        # Calculate the inverse permutation
        inverse_permutation = np.argsort(permutation)
        # Flip the required axes in reverse order
        for axis in np.where(flips)[0]:
            array = np.flip(array, axis=axis)
        # Transpose the array back to its original orientation
        original_array = np.transpose(array, inverse_permutation)
        return original_array

    def get_python_version(env_path):
        try:
            # Run the Python executable of a pyhthon environment with --version
            result = subprocess.run(
                [env_path, '-c', "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except Exception as e:
            return f"Error: {e}"

    def module_exists(module_name):
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    def runBiomedisa(
                input: vtkMRMLScalarVolumeNode,
                labels: vtkMRMLLabelMapVolumeNode,
                direction_matrix: np.array,
                parameter: BiomedisaParameter) -> list:

        # convert data
        numpyLabels = Helper.expandLabelToMatchInputImage(labels, input.GetDimensions())
        numpyImage = Helper.vtkToNumpy(input)

        print(f"Running biomedisa smart interpolation with: {parameter}")

        # unify directions if required
        numpyImage = BiomedisaLogic.unify_to_identity(numpyImage, direction_matrix)
        numpyLabels = BiomedisaLogic.unify_to_identity(numpyLabels, direction_matrix)
        uniqueLabels = np.unique(numpyLabels)

        try:
            from biomedisa_extension.config import python_path, lib_path, wsl_path
        except:
            from biomedisa_extension.config_template import python_path, lib_path, wsl_path

        # run within Slicer environment
        if python_path==None and BiomedisaLogic.module_exists("pycuda"):
            from biomedisa.interpolation import smart_interpolation
            results = smart_interpolation(
                numpyImage,
                numpyLabels,
                allaxis=parameter.allaxis,
                denoise=parameter.denoise,
                nbrw=parameter.nbrw,
                sorw=parameter.sorw,
                ignore=parameter.ignore,
                only=parameter.only,
                platform=parameter.platform)

        # run in dedicated python environment
        else:

          with tempfile.TemporaryDirectory() as temp_dir:

            # temporary file paths
            image_path = temp_dir + f'/biomedisa-image.tif'
            labels_path = temp_dir + f'/biomedisa-labels.tif'
            results_path = temp_dir + f'/final.biomedisa-image.tif'

            # save temporary data
            imwrite(image_path, numpyImage)
            imwrite(labels_path, numpyLabels)

            # Create a clean environment
            new_env = os.environ.copy()

            # adapt paths for WSL
            if os.name == "nt":
                image_path = image_path.replace('\\','/').replace('C:','/mnt/c')
                labels_path = labels_path.replace('\\','/').replace('C:','/mnt/c')

            # base command
            cmd = ["-m", "biomedisa.interpolation", image_path, labels_path]

            # append parameters on demand
            if parameter.ignore != 'none':
                cmd += [f'-i={parameter.ignore}']
            if parameter.only != 'all':
                cmd += [f'-o={parameter.only}']
            if parameter.nbrw != 10:
                cmd += [f'--nbrw={parameter.nbrw}']
            if parameter.sorw != 4000:
                cmd += [f'--sorw={parameter.sorw}']
            if parameter.allaxis:
                cmd += ['-allx']
            if parameter.denoise:
                cmd += ['-d']
            if parameter.platform:
                cmd += [f'-p={bm.platform}']

            # run interpolation on Windows using WSL
            if os.name == "nt":
                if python_path==None:
                    if os.path.exists(os.path.expanduser("~")+"/biomedisa_env/bin"):
                        python_path = (os.path.expanduser("~")+"/biomedisa_env/bin/python").replace('\\','/').replace('C:','/mnt/c')
                    else:
                        python_path = "~/biomedisa_env/bin/python"
                if wsl_path==None:
                    wsl_path = ['wsl','-e','bash','-c']
                if lib_path==None:
                    lib_path = "export CUDA_HOME=/usr/local/cuda && export LD_LIBRARY_PATH=${CUDA_HOME}/lib64 && export PATH=${CUDA_HOME}/bin:${PATH}"
                cmd = wsl_path + [lib_path + " && " + python_path + " " + (" ").join(cmd)]

            # run interpolation on Linux
            else:
                # Path to the desired Python executable
                if python_path==None:
                    if os.path.exists(os.path.expanduser("~")+"/biomedisa_env/bin/python"):
                        python_path = os.path.expanduser("~")+"/biomedisa_env/bin/python"
                        python_version = BiomedisaLogic.get_python_version(python_path)
                        lib_path = os.path.expanduser("~")+f"/biomedisa_env/lib/python{python_version}/site-package"
                    else:
                        python_path = "/usr/bin/python3"
                        python_version = BiomedisaLogic.get_python_version(python_path)
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

            # run interpolation
            subprocess.Popen(cmd, env=new_env).wait()

            # load result
            results = {}
            results['regular'] = imread(results_path)

        if results is None:
            return None

        # get results
        regular_result = results['regular']

        # restore original directions
        regular_result = BiomedisaLogic.reverse_unify_to_identity(regular_result, direction_matrix)

        return BiomedisaLogic._getBinaryLabelMaps(regular_result, direction_matrix, labels, uniqueLabels)

