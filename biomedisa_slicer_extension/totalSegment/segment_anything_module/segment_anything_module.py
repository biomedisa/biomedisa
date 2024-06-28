import logging
import os
from typing import Annotated, Optional

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'Models'))
from mouse_event_handler import MouseEventHandler
from dimension_manager import DimensionManager
#from biomedisa_slicer_extension.totalSegment.segment_anything_module.Models.mouse_event_handler import MouseEventHandler

import vtk
import numpy as np

from segment_anything import SamPredictor, sam_model_registry


from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode


#
# segment_anything_module
#


class segment_anything_module(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Segment Anything")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Matthias Fabian"] 
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
Segment Anything an easy to use, highly functioning 
For more information visit the <a href="https://github.com/facebookresearch/segment-anything">github</a>
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
The extionsion is created by an unemployed C#.NET developer who is just getting into python.
He's very sorry for all the spaghetti code.                                            
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # segment_anything_module1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="segment_anything_module",
        sampleName="segment_anything_module1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "segment_anything_module1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="segment_anything_module1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="segment_anything_module1",
    )

    # segment_anything_module2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="segment_anything_module",
        sampleName="segment_anything_module2",
        thumbnailFileName=os.path.join(iconsPath, "segment_anything_module2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="segment_anything_module2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="segment_anything_module2",
    )


#
# segment_anything_moduleParameterNode
#


@parameterNodeWrapper
class segment_anything_moduleParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# segment_anything_moduleWidget
#


class segment_anything_moduleWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

        self.red = MouseEventHandler('Red')
        self.green = MouseEventHandler('Green')
        self.yellow = MouseEventHandler('Yellow')
#        self.redWidget = self.getWidget('Red')
#        self.redView = self.getView(self.redWidget)
#        self.redInteractor = self.getInteractor(self.redView)
#
#        self.greenWidget = self.getWidget('Green')
#        self.greenView = self.getView(self.greenWidget)
#        self.greenInteractor = self.getInteractor(self.greenView)
#
#        self.yellowWidget = self.getWidget('Yellow')
#        self.yellowView = self.getView(self.yellowWidget)
#        self.yellowInteractor = self.getInteractor(self.yellowView)

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/segment_anything_module.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = segment_anything_moduleLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        # Add observer for left mouse button click
        self.red.setup()
        self.green.setup()
        self.yellow.setup()
#        self.redObserverId = self.redInteractor.AddObserver(vtkCommand.LeftButtonPressEvent, self.on_left_click_red)
#        self.greenObserverId = self.greenInteractor.AddObserver(vtkCommand.LeftButtonPressEvent, self.on_left_click_green)
#        self.yellowObserverId = self.yellowInteractor.AddObserver(vtkCommand.LeftButtonPressEvent, self.on_left_click_yellow)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()
        self.red.cleanup()
        self.green.cleanup()
        self.yellow.cleanup()
#        self.redInteractor.RemoveObserver(self.redObserverId)
#        self.greenInteractor.RemoveObserver(self.greenObserverId)
#        self.yellowInteractor.RemoveObserver(self.yellowObserverId)

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()
  
    def getWidget(self, color): # 'Red', Green Yellow
        return slicer.app.layoutManager().sliceWidget(color)
    
    def getView(self, widget):
        return widget.sliceView()
    
    def getInteractor(self, view):
        interactor = view.interactorStyle().GetInteractor()
        return interactor

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[segment_anything_moduleParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume: #TODO: Check if any points are set for index
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

            
    def onApplyButton(self) -> None:
        # TODO: make 3 different buttons to run for each dimension seperately
        sliceIndex = self.red.getCurrentIndex()
        foreground  = self.red.getForegroundCoords(sliceIndex)
        background  = self.red.getBackgroundCoords(sliceIndex)

        inputNode  =  self._parameterNode.inputVolume

        mask = self.logic.processRed(inputNode, sliceIndex, foreground, background)

        #TODO: set labels in UI
        labelsNode=slicer.util.getNode('labels')
        labelImage = labelsNode.GetImageData()
        npLabel=segment_anything_moduleLogic.vtkToNumpy(labelImage)

        self.red.dimension_manager.setSlice(npLabel, sliceIndex, mask)
        
        vtlLabel = segment_anything_moduleLogic.numpyToVTK(npLabel)
        labelsNode.SetAndObserveImageData(vtlLabel)

        #"""Run processing when user clicks "Apply" button."""
        #with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
        #    # Compute output
        #    self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
        #                       self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)
#
        #    # Compute inverted output (if needed)
        #    if self.ui.invertedOutputSelector.currentNode():
        #        # If additional output volume is selected then result with inverted threshold is written there
        #        self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
        #                           self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)

    def on_left_click_green(self, caller, event):
        coords = self.get_image_coordinates(self.redInteractor)
        print(f"green {coords}")

    def on_left_click_yellow(self, caller, event):
        coords = self.get_image_coordinates(self.redInteractor)
        print(f"yellow {coords}")

    def on_left_click_red(self, caller, event):
        coords = self.get_image_coordinates(self.redInteractor)
        modifiers = QGuiApplication.queryKeyboardModifiers()
        alt_pressed = modifiers & Qt.AltModifier
        if alt_pressed:
            # TODO: Add to non segment points - 
            print(f"no segment {coords}")
        else:
            # TODO: Add to non segment points - 
            #Add to segment points
            print(f"segment {coords}")


    def get_image_coordinates(self, interactor):
        infoWidget = slicer.modules.DataProbeInstance.infoWidget
        return self.ijkStringToCoordArray(infoWidget.layerIJKs['B'].text)
    
    def ijkStringToCoordArray(self, text):
        numbers_str = text.strip("() ").split(", ")
        numbers_uint8 = np.array(numbers_str, dtype=np.uint8)
        numbers_uint8 = numbers_uint8.reshape(-1, 1)
        return numbers_uint8
    
#
# segment_anything_moduleLogic
#


class segment_anything_moduleLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.redDimension = DimensionManager('Red')
        self.setupPredictor()

    def setupPredictor(self):
        #TODO: include file and/or make it selectable

        script_dir = os.path.dirname(os.path.abspath(__file__))
        sam_checkpoint = os.path.join(script_dir, "Resources", "sam_vit_h_4b8939.pth")
        if not os.path.exists(sam_checkpoint):
            raise Exception(f"You need to download a model checkpoint and store it at {sam_checkpoint}. Check out: https://github.com/facebookresearch/segment-anything")
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(sam)

    def getParameterNode(self):
        return segment_anything_moduleParameterNode(super().getParameterNode())

    #source: https://discourse.vtk.org/t/convert-vtk-array-to-numpy-array/3152/3
    def vtkToNumpy(data):
        from vtkmodules.util.numpy_support import vtk_to_numpy

        temp = vtk_to_numpy(data.GetPointData().GetScalars())
        dims = data.GetDimensions()
        component = data.GetNumberOfScalarComponents()
        if component == 1:
            numpy_data = temp.reshape(dims[2], dims[1], dims[0])
            numpy_data = numpy_data.transpose(2,1,0)
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
            flat_data_array = data.transpose(2,1,0).flatten()
            from vtkmodules.util.numpy_support import numpy_to_vtk
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
        img.SetDimensions(shape[0], shape[1], shape[2])
        return img
    
    def processRed(self,
                   inputVolume: vtkMRMLScalarVolumeNode,
                   index: int,
                   foreground: np.array,
                   background: np.array):
        
        print(foreground)
        print(background)

        import numpy as np
        length_f = len(foreground)
        length_b = len(background)
        point_coords = np.empty((length_f+length_b, 2), dtype=int)
        point_labels = np.empty((length_f+length_b), dtype=int)

        for i in range(length_f):
            point = foreground[i]
            point_coords[i] = [point[1], point[0]] # X and Y are inverted in comparision to Slicer
            point_labels[i] = 1
        for i in range(length_b):
            point = background[i]
            point_coords[length_f+i] = [point[1], point[0]]
            point_labels[length_f+i] = 0

        print("point_coords")
        print(point_coords)
        print("point_labels")
        print(point_labels)

        inputImage = inputVolume.GetImageData()
        npImage = segment_anything_moduleLogic.vtkToNumpy(inputImage)
        slice = self.redDimension.getSlice(npImage, index)


        import imageio

        grayscale_image = np.array(slice, dtype=np.uint8)
        imageio.imwrite(r"C:\Users\matze\Downloads\debug\grayscale_image.png", grayscale_image)
        # Stack the grayscale image into three channels to create an RGB image
        rgb_image = np.stack((grayscale_image,)*3, axis=-1)
        self.predictor.set_image(rgb_image)

        print(f"rgb_image.dtype: {rgb_image.dtype}")
        print(f"rgb_image.shape: {rgb_image.shape}")


        masks, _, _  = self.predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=False)
        masks_uint8 = (masks* 100).astype(np.uint8) #100 is the segment number

        imageio.imwrite(r"C:\Users\matze\Downloads\debug\masks_uint8_0.png", masks_uint8[0])

        return masks_uint8[0] # 0 = Take the best one


    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")        

    def runSegment():
        # TODO: make 3 seperate funtions to get the input and set the output for each dimension


        sam_checkpoint = r"C:\Users\matze\Documents\Code\biomedisa\biomedisa_slicer_extension\totalSegment\segment_anything_module\Resources\sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        imagePath=r"C:\Users\matze\Documents\Code\biomedisa\media\tumor.tif"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(sam)
        volume=slicer.util.getNode('tumor')
        inputImage = volume.GetImageData()
        npImage = segment_anything_moduleLogic.vtkToNumpy(inputImage)
        # Slice depends on which of the three images is selected. this is simply the yellow image
        slice = npImage[71] 

        import numpy as np
        # Example 2D ndarray (grayscale image)
        grayscale_image = np.array(slice, dtype=np.uint8)
        # Stack the grayscale image into three channels to create an RGB image
        rgb_image = np.stack((grayscale_image,)*3, axis=-1)
        self.predictor.set_image(rgb_image)

        #X and Y are inverted in comparision to Slicer
        point_coords = np.array([
            [30,30], [71, 172],
            [50, 120], [74, 111]])
        point_labels = np.array([
            0, 0,
            1, 1]) 
        masks, _, _  = self.predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=False)

        #vtkmask = numpyToVTK(masks)

        labelsNode=slicer.util.getNode('labels.tumor')
        labelImage = labelsNode.GetImageData()
        npLabel=segment_anything_moduleLogic.vtkToNumpy(labelImage)
        
        masks_uint8 = (masks* 100).astype(np.uint8) #100 is the segment number
        npLabel[71]=masks_uint8[0]
        vtlLabel = segment_anything_moduleLogic.numpyToVTK(npLabel)
        labelsNode.SetAndObserveImageData(vtlLabel)

#
# segment_anything_moduleTest
#

class segment_anything_moduleTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_segment_anything_module1()

    def test_segment_anything_module1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("segment_anything_module1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = segment_anything_moduleLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")


