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
from tifffile import imwrite
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
        labelMapList = []
        for label in np.unique(labelmapArray):
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
                labels: vtkMRMLLabelMapVolumeNode,
                volumeNode,
                parameter: BiomedisaPredictionParameter) -> list:
        print(f"Running biomedisa prediction with: {parameter}")

        # convert data
        if labels is not None:
            numpyLabels = Helper.expandLabelToMatchInputImage(labels, input.GetDimensions())
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
            # TODO: create segmentationNode

        # run within dedicated python environment
        else:

          with tempfile.TemporaryDirectory() as temp_dir:
            separation = False # TODO: get from model

            # temporary file paths
            image_path = os.path.join(temp_dir, 'biomedisa-image.tif')
            mask_path = os.path.join(temp_dir, 'biomedisa-mask.tif')
            extension = '.tif' if separation else '.nrrd'
            results_path = os.path.join(temp_dir, f'final.biomedisa-image{extension}')
            error_path = os.path.join(temp_dir, 'biomedisa-error.txt')
            boundary_path = results_path

            # save temporary data
            imwrite(image_path, numpyImage)
            if labels is not None:
                numpyLabels[numpyLabels>0]=1
                imwrite(mask_path, numpyLabels)

            # model path
            model_path = parameter.path_to_model

            # adapt paths for WSL
            if os.path.exists(os.path.expanduser("~")+"/anaconda3/envs/biomedisa/python.exe") and wsl_path==None:
                wsl_path=False
            if os.name == "nt" and wsl_path!=False:
                image_path = image_path.replace('\\','/').replace('C:','/mnt/c')
                model_path = model_path.replace('\\','/').replace('C:','/mnt/c')
                mask_path = model_path.replace('\\','/').replace('C:','/mnt/c')
                boundary_path = model_path.replace('\\','/').replace('C:','/mnt/c')

            # base command
            cmd = ["-m", "biomedisa.deeplearning", image_path, model_path,
                "-p", f"-ext={extension}", "--slicer"]

            # append parameters on demand
            if separation:
                if labels is None:
                    import qt
                    msgBox = qt.QMessageBox()
                    msgBox.setIcon(qt.QMessageBox.Critical)
                    msgBox.setText('Binary mask of instances required for separation model.')
                    msgBox.setWindowTitle("Error")
                    msgBox.exec_()
                    return None
                #batch_size = 1024
                parameter.stride_size = 4
                cmd += [f'-m={mask_path}', '-xp=16', '-yp=16', '-zp=16', '-s']   #TODO -s
            if parameter.stride_size != 32:
                cmd += [f'-ss={parameter.stride_size}']
            if batch_size:
                cmd += [f'-bs={batch_size}']

            # build environment
            cmd, env = Helper.build_environment(cmd)

            # run prediction
            subprocess.Popen(cmd, env=env).wait()

            # prompt error message
            if os.path.exists(error_path):
                with open(error_path, 'r') as file:
                    import qt
                    msgBox = qt.QMessageBox()
                    msgBox.setIcon(qt.QMessageBox.Critical)
                    msgBox.setText(file.read())
                    #msgBox.setInformativeText(error_message)
                    msgBox.setWindowTitle("Error")
                    msgBox.exec_()

            # load result
            if os.path.exists(results_path):
                # label particles
                if separation:
                    subprocess.Popen([cmd[0], "-m", "biomedisa.particles", image_path, mask_path, f"-bp={boundary_path}"], env=env).wait()
                    # adjust result path to particles result
                    basename = os.path.basename(results_path).replace('.tif', '.nrrd')
                    results_path = os.path.join(temp_dir, basename.replace('final.', 'result.', 1))

                # Define properties for segmentation import
                properties = {"name": "Segmentation preview", "filetype": "SegmentationFile"}

                # Load the segmentation node directly
                segmentationNode = slicer.util.loadNodeFromFile(
                    results_path,
                    "SegmentationFile",
                    properties
                )
                return segmentationNode
            else:
                return None

        #return segmentationNode#BiomedisaPredictionLogic._getBinaryLabelMaps(regular_result, volumeNode, dimensions, parameter)

