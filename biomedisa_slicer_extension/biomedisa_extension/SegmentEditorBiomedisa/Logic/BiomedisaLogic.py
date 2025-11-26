import platform
import numpy as np
import vtk, slicer
import vtk.util.numpy_support as vtk_np
from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLLabelMapVolumeNode
from Logic.BiomedisaParameter import BiomedisaParameter
from SegmentEditorCommon.Helper import Helper
import subprocess
import tempfile
from tifffile import imread, imwrite
import os

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

        try:
            from biomedisa_extension.config import python_path, wsl_path
        except:
            from biomedisa_extension.config_template import python_path, wsl_path

        # run within Slicer environment
        if python_path==None and (Helper.module_exists("pycuda") or Helper.module_exists("pyopencl")):
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

        # run within dedicated python environment
        else:

          with tempfile.TemporaryDirectory() as temp_dir:

            # temporary file paths
            image_path = os.path.join(temp_dir, 'biomedisa-image.tif')
            labels_path = os.path.join(temp_dir, 'biomedisa-labels.tif')
            results_path = os.path.join(temp_dir, 'final.biomedisa-image.tif')
            error_path = os.path.join(temp_dir, 'biomedisa-error.txt')

            # save temporary data
            imwrite(image_path, numpyImage)
            imwrite(labels_path, numpyLabels)

            # adapt paths for WSL
            if os.path.exists(os.path.expanduser("~")+"/anaconda3/envs/biomedisa/python.exe") and wsl_path==None:
                wsl_path=False
            if os.name == "nt" and wsl_path!=False:
                image_path = image_path.replace('\\','/').replace('C:','/mnt/c')
                labels_path = labels_path.replace('\\','/').replace('C:','/mnt/c')

            # base command
            cmd = ["-m", "biomedisa.interpolation", image_path, labels_path, "--slicer"]

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
                cmd += [f'-p={parameter.platform}']

            # build environment
            cmd, env = Helper.build_environment(cmd)

            # run interpolation
            subprocess.Popen(cmd, env=env).wait()

            # prompt error message
            if os.path.exists(error_path):
                with open(error_path, 'r') as file:
                    Helper.prompt_error_message(file.read())

            # load result
            results = None
            if os.path.exists(results_path):
                results = {}
                results['regular'] = imread(results_path)

        if results is None:
            return None

        # restore original directions
        regular_result = BiomedisaLogic.reverse_unify_to_identity(results['regular'], direction_matrix)

        return regular_result#BiomedisaLogic._getBinaryLabelMaps(regular_result, direction_matrix, labels, uniqueLabels)

