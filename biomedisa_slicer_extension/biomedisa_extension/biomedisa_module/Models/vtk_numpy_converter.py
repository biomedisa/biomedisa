import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.util.numpy_support import numpy_to_vtk

class vtkNumpyConverter:

    #source: https://discourse.vtk.org/t/convert-vtk-array-to-numpy-array/3152/3
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

    #source: https://discourse.vtk.org/t/convert-vtk-array-to-numpy-array/3152/3
    def numpyToVTK(data, multi_component=False, type='float'):
        '''
        multi_components: rgb has 3 components
        type：float or char
        '''
        if type == 'float':
            data_type = vtk.VTK_FLOAT
        elif type == 'char':
            data_type = vtk.VTK_UNSIGNED_CHAR
        else:
            raise RuntimeError('unknown type')
        if multi_component == False:
            if len(data.shape) == 2:
                data = data[:, :, np.newaxis]
            flat_data_array = data.transpose(0, 1, 2).flatten() # Not like in source
            vtk_data = numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)
            shape = data.shape
        else:
            assert len(data.shape) == 3, 'only test for 2D RGB'
            flat_data_array = data.transpose(1, 0, 2)
            flat_data_array = np.reshape(flat_data_array, newshape=[-1, data.shape[2]])
            vtk_data = numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)
            shape = [data.shape[0], data.shape[1], 1]
        img = vtk.vtkImageData()
        img.GetPointData().SetScalars(vtk_data)
        img.SetDimensions(shape[2], shape[1], shape[0]) # Not like in source
        return img