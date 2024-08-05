import platform
import numpy as np
import vtk, slicer
import vtk.util.numpy_support as vtk_np
from vtkmodules.util.numpy_support import vtk_to_numpy
from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLLabelMapVolumeNode
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
        #newLabelImageData = vtk.vtkImageData()
        #newLabelImageData.SetDimensions(inputDimensions)
        #newLabelImageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        # Get the bounds and extent of the original label image data
        #labelBounds = labelImageData.GetBounds()
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
        #offsets = [-labelBounds[1] + 0.5,
        #           -labelBounds[3] + 0.5,
        #           labelBounds[4] + 0.5]

        # Copy label data to the new image data at the correct position
        zmin, zmax = labelExtent[4], labelExtent[5] + 1
        ymin, ymax = labelExtent[2], labelExtent[3] + 1
        xmin, xmax = labelExtent[0], labelExtent[1] + 1
        #zminOffset, zmaxOffset = int(round(zmin-offsets[2])), int(round(zmax-offsets[2]))
        #yminOffset, ymaxOffset = int(round(ymin-offsets[1])), int(round(ymax-offsets[1]))
        #xminOffset, xmaxOffset = int(round(xmin-offsets[0])), int(round(xmax-offsets[0]))
        newLabelNumpyArray[zmin:zmax, ymin:ymax, xmin:xmax] = labelNumpyArray
        #= labelNumpyArray[zminOffset:zmaxOffset, yminOffset:ymaxOffset, xminOffset:xmaxOffset]

        # Convert the NumPy array back to a VTK array and set it as the scalars of the new VTK image data object
        #newLabelVtkArray = vtk_np.numpy_to_vtk(newLabelNumpyArray.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        #newLabelImageData.GetPointData().SetScalars(newLabelVtkArray)

        return newLabelNumpyArray

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
        extendedLabel = BiomedisaLogic._expandLabelToMatchInputImage(labels, input.GetDimensions())
        numpyLabels = BiomedisaLogic._vtkToNumpy(extendedLabel)
        from biomedisa.interpolation import read_labeled_slices
        labeledSlices, _ = read_labeled_slices(numpyLabels)
        return labeledSlices

    def unify_to_identity(array, direction_matrix):
        # Determine the permutation of axes
        permutation = np.argmax(np.abs(direction_matrix), axis=0)
        print(permutation)
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
        numpyLabels = BiomedisaLogic._expandLabelToMatchInputImage(labels, input.GetDimensions())
        #numpyLabels = BiomedisaLogic._vtkToNumpy(extendedLabel)
        numpyImage = BiomedisaLogic._vtkToNumpy(input)

        print(f"Running biomedisa smart interpolation with: {parameter}")

        # unify directions if required
        numpyImage = BiomedisaLogic.unify_to_identity(numpyImage, direction_matrix)
        numpyLabels = BiomedisaLogic.unify_to_identity(numpyLabels, direction_matrix)
        uniqueLabels = np.unique(numpyLabels)

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

        if results is None:
            return None

        # get results
        regular_result = results['regular']

        # restore original directions
        regular_result = BiomedisaLogic.reverse_unify_to_identity(regular_result, direction_matrix)

        return BiomedisaLogic._getBinaryLabelMaps(regular_result, direction_matrix, labels, uniqueLabels)

