import numpy as np
import vtk, slicer
import vtk.util.numpy_support as vtk_np
from vtkmodules.util.numpy_support import vtk_to_numpy
from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLLabelMapVolumeNode
from biomedisa_extension.SegmentEditorBiomedisaPrediction.Logic.BiomedisaPredictionParameter import BiomedisaPredictionParameter
from biomedisa_extension.SegmentEditorCommon.Helper import Helper

class BiomedisaPredictionLogic():

    #source: https://discourse.vtk.org/t/convert-vtk-array-to-numpy-array/3152/3
    def _vtkToNumpy(data):
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
        
        numpyImage = BiomedisaPredictionLogic._vtkToNumpy(input)
        dimensions = input.GetDimensions()
        print(f"numpyImage: {numpyImage.shape}")
        numpyImage = Helper.crop(numpyImage, dimensions, parameter.x_min, parameter.x_max, parameter.y_min, parameter.y_max, parameter.z_min, parameter.z_max)
        print(f"Running biomedisa prediction with: {parameter}")

        print(f"dimensions: {dimensions}")
        print(f"crop: {numpyImage.shape}")
        
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

        if False:
            import debugpy
            print("Waiting for debugger attach...")
            debugpy.wait_for_client()
            print("Debugger attached, continuing execution...")

        regular_result = results['regular']

        labelmapList = BiomedisaPredictionLogic._getBinaryLabelMaps(regular_result, volumeNode, dimensions, parameter)

        return labelmapList

