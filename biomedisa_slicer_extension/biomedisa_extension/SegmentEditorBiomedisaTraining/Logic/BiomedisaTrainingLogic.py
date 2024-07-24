import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np
from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLLabelMapVolumeNode
from SegmentEditorCommon.Helper import Helper

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
    
    def trainDeepLearning(
            input: vtkMRMLScalarVolumeNode,
            labels: vtkMRMLLabelMapVolumeNode, 
            modelFile: str,
            stride_size: int,
            epochs: int,
            validation_split: float,
            balance: bool,
            swapaxes: bool,
            flip_x: bool, flip_y:bool, flip_z:bool,
            scaling: bool,
            x_scale: int, y_scale:int, z_scale:int):
        
        extendedLabel = BiomedisaTrainingLogic._expandLabelToMatchInputImage(labels, input.GetDimensions())
        numpyLabels = Helper._vtkToNumpy(extendedLabel)
        numpyImage = Helper._vtkToNumpy(input)

        from biomedisa.deeplearning import deep_learning
        deep_learning(
            img_data=numpyImage, 
            label_data=numpyLabels,
            path_to_model=str(modelFile),
            train=True,
            stride_size=stride_size,
            epochs=epochs,
            validation_split=validation_split,
            balance=balance,
            swapaxes=swapaxes,
            flip_x=flip_x,
            flip_y=flip_y,
            flip_z=flip_z,
            scaling=scaling,
            x_scale=x_scale, 
            y_scale=y_scale, 
            z_scale=z_scale)
