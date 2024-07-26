import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np
from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLLabelMapVolumeNode
from SegmentEditorCommon.Helper import Helper
from biomedisa_extension.SegmentEditorBiomedisaTraining.Logic.BiomedisaTrainingParameter import BiomedisaTrainingParameter

class BiomedisaTrainingLogic():

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

        # Copy label data to the new image data at the correct position
        zmin, zmax = labelExtent[4], labelExtent[5] + 1
        ymin, ymax = labelExtent[2], labelExtent[3] + 1
        xmin, xmax = labelExtent[0], labelExtent[1] + 1
        zminOffset, zmaxOffset = int(round(zmin-offsets[2])), int(round(zmax-offsets[2]))
        yminOffset, ymaxOffset = int(round(ymin-offsets[1])), int(round(ymax-offsets[1]))
        xminOffset, xmaxOffset = int(round(xmin-offsets[0])), int(round(xmax-offsets[0]))
        newLabelNumpyArray[zmin:zmax, ymin:ymax, xmin:xmax] \
            = labelNumpyArray[zminOffset:zmaxOffset, yminOffset:ymaxOffset, xminOffset:xmaxOffset]

        # Convert the NumPy array back to a VTK array and set it as the scalars of the new VTK image data object
        newLabelVtkArray = vtk_np.numpy_to_vtk(newLabelNumpyArray.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        newLabelImageData.GetPointData().SetScalars(newLabelVtkArray)
        return newLabelImageData
    
    def trainDeepLearning(
            input: vtkMRMLScalarVolumeNode,
            labels: vtkMRMLLabelMapVolumeNode, 
            parameter: BiomedisaTrainingParameter):
        
        extendedLabel = BiomedisaTrainingLogic._expandLabelToMatchInputImage(labels, input.GetDimensions())
        numpyLabels = Helper._vtkToNumpy(extendedLabel)
        numpyImage = Helper._vtkToNumpy(input)

        print(f"Running biomedisa training with: {parameter}")

        from biomedisa.deeplearning import deep_learning
        deep_learning(
            img_data=numpyImage, 
            label_data=numpyLabels,
            path_to_model=parameter.path_to_model,
            stride_size=parameter.stride_size,
            epochs=parameter.epochs,
            validation_split=parameter.validation_split,
            balance=parameter.balance,
            swapaxes=parameter.swapaxes,
            flip_x=parameter.flip_x,
            flip_y=parameter.flip_y,
            flip_z=parameter.flip_z,
            scaling=parameter.scaling,
            x_scale=parameter.x_scale, 
            y_scale=parameter.y_scale, 
            z_scale=parameter.z_scale,
            train=True)
