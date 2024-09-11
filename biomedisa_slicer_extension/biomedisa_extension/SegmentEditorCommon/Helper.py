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
        # Get extent of the original label image data
        labelExtent = labelImageData.GetExtent()

        # Convert the label image data to a NumPy array
        labelPointData = labelImageData.GetPointData()
        labelVtkArray = labelPointData.GetScalars()
        labelNumpyArray = vtk_np.vtk_to_numpy(labelVtkArray)
        labelNumpyArray = labelNumpyArray.reshape(labelImageData.GetDimensions()[::-1])

        # Initialize the NumPy array for the new label image data with zeros
        newLabelNumpyArray = np.zeros(inputDimensions, dtype=np.uint8)
        newLabelNumpyArray = newLabelNumpyArray.reshape(inputDimensions[::-1])

        # Copy label data to the new image data at the correct position
        zmin, zmax = labelExtent[4], labelExtent[5] + 1
        ymin, ymax = labelExtent[2], labelExtent[3] + 1
        xmin, xmax = labelExtent[0], labelExtent[1] + 1
        newLabelNumpyArray[zmin:zmax, ymin:ymax, xmin:xmax] = labelNumpyArray

        return newLabelNumpyArray

    @staticmethod
    def crop(image: np.ndarray,
             x_min: int, x_max: int,
             y_min: int, y_max: int,
             z_min: int, z_max: int) -> np.ndarray:
        image = image[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1].copy()
        return image

    @staticmethod
    def embed(image: np.ndarray,
             dimensions,
             x_min: int, x_max: int,
             y_min: int, y_max: int,
             z_min: int, z_max: int) -> np.ndarray:
        fullimage = np.zeros((dimensions[2], dimensions[1], dimensions[0]), dtype=image.dtype)
        fullimage[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1] = image
        return fullimage

