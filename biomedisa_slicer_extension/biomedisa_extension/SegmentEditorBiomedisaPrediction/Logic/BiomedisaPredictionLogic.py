import numpy as np
import vtk, slicer
import vtk.util.numpy_support as vtk_np
from vtkmodules.util.numpy_support import vtk_to_numpy
from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLLabelMapVolumeNode
from biomedisa_extension.SegmentEditorBiomedisaPrediction.Logic.BiomedisaPredictionParameter import BiomedisaPredictionParameter
from biomedisa_extension.SegmentEditorCommon.Helper import Helper
import subprocess
import tempfile
from tifffile import imread, imwrite
import os

class BiomedisaPredictionLogic():

    def _getBinaryLabelMap(label: np.array, volumeNode) -> vtkMRMLLabelMapVolumeNode:
        vtkImageData = slicer.vtkOrientedImageData()
        vtkImageData.SetDimensions(label.shape[2], label.shape[1], label.shape[0])
        direction_matrix = np.zeros((3,3))
        volumeNode.GetIJKToRASDirections(direction_matrix)
        vtkImageData.SetDirections(direction_matrix)
        vtkImageData.SetSpacing(volumeNode.GetSpacing())
        vtkImageData.SetOrigin(volumeNode.GetOrigin())
        vtkImageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        vtkArray = vtk_np.numpy_to_vtk(num_array=label.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtkImageData.GetPointData().SetScalars(vtkArray)
        return vtkImageData

    def _getBinaryLabelMaps(labelmapArray: np.array,
                            volumeNode,
                            dimensions,
                            parameter: BiomedisaPredictionParameter) -> list:
        uniqueLabels = np.unique(labelmapArray)
        labelMapList = []
        for label in uniqueLabels:
            if label == 0:
                continue
            binaryLabelmapArray = np.where(labelmapArray == label, 1, 0).astype(np.uint8)
            print(f"binaryLabelmapArray: {binaryLabelmapArray.shape}")
            binaryLabelmapArray = Helper.embed(binaryLabelmapArray, dimensions, parameter.x_min, parameter.x_max, parameter.y_min, parameter.y_max, parameter.z_min, parameter.z_max)
            print(f"uncropped: {binaryLabelmapArray.shape}")
            vtkBinaryLabelmap  = BiomedisaPredictionLogic._getBinaryLabelMap(binaryLabelmapArray, volumeNode)
            labelMapList.append((int(label), vtkBinaryLabelmap))

        return labelMapList

    def predictDeepLearning(
                input: vtkMRMLScalarVolumeNode,
                volumeNode,
                parameter: BiomedisaPredictionParameter) -> list:
        print(f"Running biomedisa prediction with: {parameter}")

        numpyImage = Helper.vtkToNumpy(input)
        dimensions = input.GetDimensions()
        numpyImage = Helper.crop(numpyImage, parameter.x_min, parameter.x_max, parameter.y_min, parameter.y_max, parameter.z_min, parameter.z_max)

        # batch size
        batch_size = parameter.batch_size if parameter.batch_size_active else None

        try:
            from biomedisa_extension.config import python_path, wsl_path
        except:
            from biomedisa_extension.config_template import python_path, wsl_path

        # run within Slicer environment
        if python_path==None and Helper.module_exists("tensorflow"):
            from biomedisa.deeplearning import deep_learning
            results = deep_learning(numpyImage,
                                    path_to_model=parameter.path_to_model,
                                    stride_size=parameter.stride_size,
                                    batch_size=batch_size,
                                    predict=True)

        # run within dedicated python environment
        else:

          with tempfile.TemporaryDirectory() as temp_dir:

            # temporary file paths
            image_path = temp_dir + '/biomedisa-image.tif'
            results_path = temp_dir + '/final.biomedisa-image.tif'

            # save temporary data
            imwrite(image_path, numpyImage)

            # model path
            model_path = parameter.path_to_model

            # adapt paths for WSL
            if os.name == "nt" and wsl_path!=False:
                image_path = image_path.replace('\\','/').replace('C:','/mnt/c')
                model_path = model_path.replace('\\','/').replace('C:','/mnt/c')

            # base command
            cmd = ["-m", "biomedisa.deeplearning", image_path, model_path, '-ext', '.tif']

            # append parameters on demand
            if parameter.stride_size != 32:
                cmd += [f'-ss={parameter.stride_size}']
            if batch_size:
                cmd += [f'-bs={batch_size}']

            # build environment
            cmd, env = Helper.build_environment(cmd)

            # run prediction
            subprocess.Popen(cmd, env=env).wait()

            # load result
            results = {}
            results['regular'] = imread(results_path)

        if results is None:
            print("No result")
            return None

        # get results
        regular_result = results['regular']

        return BiomedisaPredictionLogic._getBinaryLabelMaps(regular_result, volumeNode, dimensions, parameter)

