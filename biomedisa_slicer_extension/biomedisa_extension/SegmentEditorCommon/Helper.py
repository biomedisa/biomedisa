import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np
from vtkmodules.util.numpy_support import vtk_to_numpy

class Helper():

    #source: https://discourse.vtk.org/t/convert-vtk-array-to-numpy-array/3152/3
    @staticmethod
    def vtkToNumpy(data):
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

    @staticmethod
    def expandLabelToMatchInputImage(labelImageData, inputDimensions) -> vtk.vtkImageData:
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
    
    @staticmethod
    def crop(image: np.ndarray, 
             dimensions,
             x_min: int, x_max: int, 
             y_min: int, y_max: int, 
             z_min: int, z_max: int) -> np.ndarray:
        
        x_range = [x_min, x_max] if x_min > 0 or dimensions[0]-1 > x_max else None
        y_range = [y_min, y_max] if y_min > 0 or dimensions[1]-1 > y_max else None
        z_range = [z_min, z_max] if z_min > 0 or dimensions[2]-1 > z_max else None
        if z_range is not None:
            image = image[z_range[0]:z_range[1]+1].copy()
        if y_range is not None:
            image = image[:,y_range[0]:y_range[1]+1].copy()
        if x_range is not None:
            image = image[:,:,x_range[0]:x_range[1]+1].copy()
        return image

    @staticmethod
    def embed(image: np.ndarray, 
             dimensions,
             x_min: int, x_max: int, 
             y_min: int, y_max: int, 
             z_min: int, z_max: int) -> np.ndarray:
        
        x_range = [x_min, x_max] if x_min > 0 or dimensions[0]-1 > x_max else None
        y_range = [y_min, y_max] if y_min > 0 or dimensions[1]-1 > y_max else None
        z_range = [z_min, z_max] if z_min > 0 or dimensions[2]-1 > z_max else None
        if x_range is None and y_range is None and z_range is None:
            return image

        fullimage = np.zeros((dimensions[2], dimensions[1], dimensions[0]))
        print(f"fullimage: {fullimage.shape}")
        print(f"x_range: {x_range}")
        print(f"y_range: {y_range}")
        print(f"z_range: {z_range}")
        if x_range is not None and y_range is not None and z_range is not None:
            print(f"copy")
            fullimage[z_range[0]:z_range[1]+1, y_range[0]:y_range[1]+1, x_range[0]:x_range[1]+1] = image.copy()
            print(f"copied")
        elif x_range is not None and z_range is not None:
            fullimage[z_range[0]:z_range[1]+1, :, x_range[0]:x_range[1]+1] = image.copy()
        elif y_range is not None and z_range is not None:
            fullimage[z_range[0]:z_range[1]+1, y_range[0]:y_range[1]+1, :] = image.copy()
        elif x_range is not None and y_range is not None:
            fullimage[:, y_range[0]:y_range[1]+1, x_range[0]:x_range[1]+1] = image.copy()
        elif x_range is not None:
            fullimage[:, :, x_range[0]:x_range[1]+1] = image.copy()
        elif y_range is not None:
            fullimage[:, y_range[0]:y_range[1]+1, :] = image.copy()
        elif z_range is not None:
            fullimage[z_range[0]:z_range[1]+1, :, :] = image.copy()
    
        return fullimage