import numpy as np
import vtk, slicer
import vtk.util.numpy_support as vtk_np
from vtkmodules.util.numpy_support import vtk_to_numpy
from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLLabelMapVolumeNode
from biomedisa.interpolation import smart_interpolation
from Logic.BiomedisaParameter import BiomedisaParameter

class BiomedisaLogic():

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

    def _expandLabelToMatchInputImage(labelImageData, inputDimensions) -> vtk.vtkImageData:
        # Initialize the new VTK image data object with the same dimensions as the input image
        newLabelImageData = vtk.vtkImageData()
        newLabelImageData.SetDimensions(inputDimensions)
        newLabelImageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # Get the bounds and extent of the original label image data
        labelBounds = labelImageData.GetBounds()
        labelExtent = labelImageData.GetExtent()
        
        # Convert the label image data to a NumPy array
        labelPointData = labelImageData.GetPointData()
        labelVtkArray = labelPointData.GetScalars()
        labelNumpyArray = vtk_np.vtk_to_numpy(labelVtkArray)
        labelNumpyArray = labelNumpyArray.reshape(labelImageData.GetDimensions()[::-1])

        # Initialize the NumPy array for the new label image data with zeros
        newLabelNumpyArray = np.zeros(inputDimensions, dtype=np.uint8)
        newLabelNumpyArray = newLabelNumpyArray.reshape(inputDimensions[::-1])

        # Calculate the offset for copying the label data to the correct position in the new image
        offsets = [-labelBounds[1] + 0.5,
                   -labelBounds[3] + 0.5,
                   labelBounds[4] + 0.5]

        # Iterate over the label data and copy it to the new image data at the correct position
        for z in range(labelExtent[4], labelExtent[5] + 1):
            for y in range(labelExtent[2], labelExtent[3] + 1):
                for x in range(labelExtent[0], labelExtent[1] + 1):
                    zz = int(z - offsets[2])
                    yy = int(y - offsets[1])
                    xx = int(x - offsets[0])
                    newLabelNumpyArray[z, y, x] = labelNumpyArray[zz, yy, xx]

        # Convert the NumPy array back to a VTK array and set it as the scalars of the new VTK image data object
        newLabelVtkArray = vtk_np.numpy_to_vtk(newLabelNumpyArray.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        newLabelImageData.GetPointData().SetScalars(newLabelVtkArray)

        return newLabelImageData

    def _getBinaryLabelMap(label: np.array) -> vtkMRMLLabelMapVolumeNode:
        vtkImageData = slicer.vtkOrientedImageData()
        vtkImageData.SetDimensions(label.shape[2], label.shape[1], label.shape[0])
        vtkImageData.SetDirections([[-1,0,0],[0,-1,0],[0,0,1]])
        vtkImageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        vtkArray = vtk_np.numpy_to_vtk(num_array=label.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtkImageData.GetPointData().SetScalars(vtkArray)
        return vtkImageData

    def _getBinaryLabelMaps(labelmapArray: np.array) -> list:
        uniqueLabels = np.unique(labelmapArray)
        labelMapList = []
        for label in uniqueLabels:
            if label == 0:
                continue
            binaryLabelmapArray = np.where(labelmapArray == label, 1, 0).astype(np.uint8)
            vtkBinaryLabelmap  = BiomedisaLogic._getBinaryLabelMap(binaryLabelmapArray)
            print(f"label: {label}")
            labelMapList.append((int(label), vtkBinaryLabelmap))

        return labelMapList
    
    def runBiomedisa(
                input: vtkMRMLScalarVolumeNode,
                labels: vtkMRMLLabelMapVolumeNode, 
                parameter: BiomedisaParameter = None) -> list:

        # convert data
        extendedLabel = BiomedisaLogic._expandLabelToMatchInputImage(labels, input.GetDimensions())
        numpyImage = BiomedisaLogic._vtkToNumpy(input)
        numpyLabels = BiomedisaLogic._vtkToNumpy(extendedLabel)

        if(parameter is None):
            results = smart_interpolation(numpyImage, numpyLabels)
        else:
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
            
        if results is None:
            return None
        
        # get results
        regular_result = results['regular']

        return BiomedisaLogic._getBinaryLabelMaps(regular_result)
