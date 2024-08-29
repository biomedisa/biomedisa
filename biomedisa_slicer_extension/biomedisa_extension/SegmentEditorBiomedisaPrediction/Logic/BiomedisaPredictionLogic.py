import numpy as np
import vtk, slicer
import vtk.util.numpy_support as vtk_np
from vtkmodules.util.numpy_support import vtk_to_numpy
from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLLabelMapVolumeNode
from biomedisa_extension.SegmentEditorBiomedisaPrediction.Logic.BiomedisaPredictionParameter import BiomedisaPredictionParameter
from biomedisa_extension.SegmentEditorCommon.Helper import Helper

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

        batch_size = parameter.batch_size if parameter.batch_size_active else None
        from biomedisa.deeplearning import deep_learning
        results = deep_learning(numpyImage, 
                                path_to_model=parameter.path_to_model, 
                                stride_size=parameter.stride_size, 
                                batch_size=batch_size, 
                                predict=True)
        if results is None:
            print("No result")
            return None

        regular_result = results['regular']

        labelmapList = BiomedisaPredictionLogic._getBinaryLabelMaps(regular_result, volumeNode, dimensions, parameter)

        return labelmapList

