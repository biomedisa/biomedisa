import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np
from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLLabelMapVolumeNode
from SegmentEditorCommon.Helper import Helper
from biomedisa_extension.SegmentEditorBiomedisaTraining.Logic.BiomedisaTrainingParameter import BiomedisaTrainingParameter
import subprocess
import tempfile
from tifffile import imwrite
import os

class BiomedisaTrainingLogic():

    def trainDeepLearning(
            input: vtkMRMLScalarVolumeNode,
            labels: vtkMRMLLabelMapVolumeNode,
            parameter: BiomedisaTrainingParameter):

        print(f"Running biomedisa training with: {parameter}")

        numpyLabels = Helper.expandLabelToMatchInputImage(labels, input.GetDimensions())
        numpyImage = Helper.vtkToNumpy(input)

        # crop training data
        numpyImage = Helper.crop(numpyImage, parameter.x_min, parameter.x_max, parameter.y_min, parameter.y_max, parameter.z_min, parameter.z_max)
        numpyLabels = Helper.crop(numpyLabels, parameter.x_min, parameter.x_max, parameter.y_min, parameter.y_max, parameter.z_min, parameter.z_max)

        try:
            from biomedisa_extension.config import python_path, wsl_path
        except:
            from biomedisa_extension.config_template import python_path, wsl_path

        # run within Slicer environment
        if python_path==None and Helper.module_exists("tensorflow"):
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

        # run within dedicated python environment
        else:

          with tempfile.TemporaryDirectory() as temp_dir:

            # temporary file paths
            image_path = temp_dir + '/biomedisa-image.tif'
            labels_path = temp_dir + '/biomedisa-labels.tif'
            error_path = temp_dir + '/biomedisa-error.txt'

            # save temporary data
            imwrite(image_path, numpyImage)
            imwrite(labels_path, numpyLabels)

            # model path
            model_path = parameter.path_to_model

            # adapt paths for WSL
            if os.name == "nt" and wsl_path!=False:
                image_path = image_path.replace('\\','/').replace('C:','/mnt/c')
                labels_path = labels_path.replace('\\','/').replace('C:','/mnt/c')
                model_path = model_path.replace('\\','/').replace('C:','/mnt/c')

            # base command
            cmd = ["-m", "biomedisa.deeplearning", image_path, labels_path,
                "-t", "-ptm", model_path, "--slicer"]

            # append parameters on demand
            if parameter.stride_size != 32:
                cmd += [f'-ss={parameter.stride_size}']
            if parameter.epochs != 100:
                cmd += [f'-e={parameter.epochs}']
            if parameter.validation_split > 0:
                cmd += [f'-vs={parameter.validation_split}']
            if parameter.balance:
                cmd += ['-b']
            if parameter.swapaxes:
                cmd += ['-sa']
            if parameter.flip_x:
                cmd += ['--flip_x']
            if parameter.flip_y:
                cmd += ['--flip_y']
            if parameter.flip_z:
                cmd += ['--flip_z']
            if parameter.scaling:
                cmd += [f'-xs={parameter.x_scale}', f'-ys={parameter.y_scale}', f'-zs={parameter.z_scale}']
            else:
                cmd += ['-ns']

            # build environment
            cmd, env = Helper.build_environment(cmd)

            # run training
            subprocess.Popen(cmd, env=env).wait()

            # prompt error message
            if os.path.exists(error_path):
                with open(error_path, 'r') as file:
                    import qt
                    msgBox = qt.QMessageBox()
                    msgBox.setIcon(qt.QMessageBox.Critical)
                    msgBox.setText(file.read())
                    #msgBox.setInformativeText(error_message)
                    msgBox.setWindowTitle("Error")
                    msgBox.exec_()

