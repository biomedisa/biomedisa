import platform
import numpy as np
import vtk, slicer
import vtk.util.numpy_support as vtk_np
from vtkmodules.util.numpy_support import vtk_to_numpy
from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLLabelMapVolumeNode
from Logic.BiomedisaParameter import BiomedisaParameter
from SegmentEditorCommon.Helper import Helper

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

