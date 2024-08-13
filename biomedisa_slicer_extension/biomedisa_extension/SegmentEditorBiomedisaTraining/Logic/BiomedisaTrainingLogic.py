import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np
from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLLabelMapVolumeNode
from SegmentEditorCommon.Helper import Helper
from biomedisa_extension.SegmentEditorBiomedisaTraining.Logic.BiomedisaTrainingParameter import BiomedisaTrainingParameter

class BiomedisaTrainingLogic():

    def trainDeepLearning(
            input: vtkMRMLScalarVolumeNode,
            labels: vtkMRMLLabelMapVolumeNode,
            parameter: BiomedisaTrainingParameter):

        numpyLabels = Helper.expandLabelToMatchInputImage(labels, input.GetDimensions())
        numpyImage = Helper.vtkToNumpy(input)

        # crop training data
        numpyImage = Helper.crop(numpyImage, parameter.x_min, parameter.x_max, parameter.y_min, parameter.y_max, parameter.z_min, parameter.z_max)
        numpyLabels = Helper.crop(numpyLabels, parameter.x_min, parameter.x_max, parameter.y_min, parameter.y_max, parameter.z_min, parameter.z_max)

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

