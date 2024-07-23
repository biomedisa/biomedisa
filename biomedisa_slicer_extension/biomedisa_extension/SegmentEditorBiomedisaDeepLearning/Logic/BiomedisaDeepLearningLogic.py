import numpy as np
import vtk, slicer
import vtk.util.numpy_support as vtk_np
from vtkmodules.util.numpy_support import vtk_to_numpy
from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLLabelMapVolumeNode

class BiomedisaDeepLearningLogic():

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

    def _getBinaryLabelMap(label: np.array, input: vtkMRMLScalarVolumeNode) -> vtkMRMLLabelMapVolumeNode:
        vtkImageData = slicer.vtkOrientedImageData()
        spacing = input.GetSpacing()
        origin = input.GetOrigin()
        directions = np.zeros((3,3))
        input.GetDirections(directions)
        vtkImageData.SetDimensions(label.shape[2], label.shape[1], label.shape[0])
        vtkImageData.SetDirections(directions)
        vtkImageData.SetSpacing(spacing)
        vtkImageData.SetOrigin(origin)
        vtkImageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        vtkArray = vtk_np.numpy_to_vtk(num_array=label.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtkImageData.GetPointData().SetScalars(vtkArray)
        return vtkImageData

    def _getBinaryLabelMaps(labelmapArray: np.array,
                            input: vtkMRMLScalarVolumeNode) -> list:
        uniqueLabels = np.unique(labelmapArray)
        labelMapList = []
        for label in uniqueLabels:
            if label == 0:
                continue
            binaryLabelmapArray = np.where(labelmapArray == label, 1, 0).astype(np.uint8)
            vtkBinaryLabelmap  = BiomedisaDeepLearningLogic._getBinaryLabelMap(binaryLabelmapArray, input)
            labelMapList.append((int(label), vtkBinaryLabelmap))

        return labelMapList

    def predictDeepLearning(
                input: vtkMRMLScalarVolumeNode,
                modelFile: str,
                stride_size: int = 32
                ) -> list:
        numpyImage = BiomedisaDeepLearningLogic._vtkToNumpy(input)

        from biomedisa.deeplearning import deep_learning
        results = deep_learning(numpyImage, path_to_model=modelFile, stride_size=stride_size, predict=True)
        if results is None:
            return None

        regular_result = results['regular']

        return BiomedisaDeepLearningLogic._getBinaryLabelMaps(regular_result, input)