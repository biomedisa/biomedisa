import logging
import os
from typing import Annotated, Optional

import numpy as np
import vtk
from vtk import vtkCommand, vtkInteractorStyleUser
from segment_anything import SamPredictor, sam_model_registry
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'Models'))
from vtk_numpy_converter import vtkNumpyConverter

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer import vtkMRMLLabelMapVolumeNode
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode


#
# biomedisa_module
#


class biomedisa_module(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Biomedisa Label")
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Matthias Fabian"]
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
Biomedisa is a free and easy-to-use open-source application for segmenting large volumetric images such as CT and MRI scans,
developed at The Australian National University CTLab. Biomedisa's smart interpolation of sparsely pre-segmented slices
enables accurate semi-automated segmentation by considering the complete underlying image data. 
For more information visit the <a href="https://biomedisa.info/">project page</a>
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

    # biomedisa_module1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="biomedisa_module",
        sampleName="biomedisa_module1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "biomedisa_module1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="biomedisa_module1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="biomedisa_module1",
    )

    # biomedisa_module2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="biomedisa_module",
        sampleName="biomedisa_module2",
        thumbnailFileName=os.path.join(iconsPath, "biomedisa_module2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="biomedisa_module2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="biomedisa_module2",
    )


#
# biomedisa_moduleParameterNode
#


@parameterNodeWrapper
class biomedisa_moduleParameterNode:
    """
    The parameters needed by module.

    """

    inputVolume: vtkMRMLScalarVolumeNode
    inputLabels: vtkMRMLLabelMapVolumeNode
    outputLabels: vtkMRMLLabelMapVolumeNode

    nbrw: int = 10
    sorw: int = 4000


#
# biomedisa_moduleWidget
#


class biomedisa_moduleWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/biomedisa_module.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.labelSelector.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = biomedisa_moduleLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.setup_markups()

        # Add observer for left mouse button click
        self.interactor.AddObserver(vtkCommand.LeftButtonPressEvent, self.on_left_click)
        
        # Buttons
        self.ui.biomedisaButton.connect("clicked(bool)", self.onBiomedisaButton)
        self.ui.segmentAnythingButton.connect("clicked(bool)", self.onSegmentAnythingButton)
        self.ui.deleteLabelButton.connect("clicked(bool)", self.onDeleteLabelButton)
        self.ui.clearPointsButton.connect("clicked(bool)", self.onClearPointsButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def setup_markups(self):
        self.slice_widget = slicer.app.layoutManager().sliceWidget('Red')
        self.slice_view = self.slice_widget.sliceView()
        self.observerId = self.interactor = self.slice_view.interactorStyle().GetInteractor()

        self.foregroundMarkupsNode = self.create_markups(self.slice_widget, "Segment", 0, 1, 0)
        self.backgroundMarkupsNode = self.create_markups(self.slice_widget, "Non Segement", 1, 0, 0)

    def create_markups(self ,slice_widget, name, r, g, b):
        # Create markups node
        markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        markupsNode.SetName(name)
        
        # Create display node
        markupsDisplayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsDisplayNode")
        markupsNode.SetAndObserveDisplayNodeID(markupsDisplayNode.GetID())
        #Move to 'Red'
        redSliceNode = slice_widget.mrmlSliceNode()
        displayNode = markupsNode.GetDisplayNode()
        displayNode.AddViewNodeID(redSliceNode.GetID())

        # Configure display node
        markupsDisplayNode.SetGlyphScale(2.0)  # Size of the fiducial points
        markupsDisplayNode.SetSelectedColor(r, g, b)  
        markupsDisplayNode.SetColor(r, g, b)  
        #markupsDisplayNode.SetColor(0, 0, 1)  

        return markupsNode

    def on_left_click(self, caller, event):
        xy = self.interactor.GetEventPosition()

        sliceValue = self.get_slice_index()

        ras = self.slice_view.convertXYZToRAS([xy[0], xy[1], sliceValue])
        ras = [ras[0], ras[1], sliceValue]

        # TODO: Check if out of frame and return
        # TODO: Check if point is selected and remove it.
       
        modifiers = QGuiApplication.queryKeyboardModifiers()
        alt_pressed = bool(modifiers & Qt.Modifier.ALT)
        if(alt_pressed):
            self.backgroundMarkupsNode.AddControlPoint(ras[0], ras[1], ras[2])
        else:
            self.foregroundMarkupsNode.AddControlPoint(ras[0], ras[1], ras[2])

    def get_slice_index(self )-> int: 
        sliceController = self.slice_widget.sliceController()
        sliceValue = sliceController.sliceOffsetSlider().value
        return int(sliceValue)
        
    def get_foreground_points(self, sliceIndex):
        return self.get_points(self.foregroundMarkupsNode, sliceIndex)

    def get_background_points(self, sliceIndex):
        return self.get_points(self.foregroundMarkupsNode, sliceIndex)

    def get_points(self, markupsNode, sliceIndex):
        points = []
        pointsCount = markupsNode.GetNumberOfControlPoints()
        for i in range(pointsCount):
            coords = [0.0, 0.0, 0.0]
            markupsNode.GetNthControlPointPosition(i, coords)
            if(coords[2] == sliceIndex):
                points.append([-round(coords[0]), -round(coords[1]), coords[2], i])
        return points

    def clear_background_points(self, sliceIndex):
        self.clear_points(self.backgroundMarkupsNode, sliceIndex)

    def clear_foreground_points(self, sliceIndex):
        self.clear_points(self.foregroundMarkupsNode, sliceIndex)

    def clear_points(self, markupsNode, sliceIndex):
        sliceIndex = self.get_slice_index()
        points = self.get_points(markupsNode, sliceIndex)
        for point in reversed(points):
            markupsNode.RemoveNthControlPoint(point[3])

    def clear_all_points(self):
        self.foregroundMarkupsNode.RemoveAllControlPoints()
        self.backgroundMarkupsNode.RemoveAllControlPoints()
    
    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()
        self.interactor.RemoveObserver(self.observerId)
        self.clear_all_points()

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
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanRunBiomedisa)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanDeleteLabel)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
            for node in nodes:
                if not isinstance(node, vtkMRMLLabelMapVolumeNode):
                    self._parameterNode.inputVolume = node
                    break

        if not self._parameterNode.inputLabels:
            nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
            for node in nodes:
                if isinstance(node, vtkMRMLLabelMapVolumeNode):
                    self._parameterNode.inputLabels = node
                    break

        if not self._parameterNode.outputLabels:
            nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
            for node in nodes:
                if isinstance(node, vtkMRMLLabelMapVolumeNode):
                    self._parameterNode.outputLabels = node
                    break

    def setParameterNode(self, inputParameterNode: Optional[biomedisa_moduleParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanRunBiomedisa)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanRunSegmentAnything)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanDeleteLabel)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanRunBiomedisa)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanRunSegmentAnything)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanDeleteLabel)
            self._checkCanRunBiomedisa()
            self._checkCanRunSegmentAnything()
            self._checkCanDeleteLabel()

    def _checkCanRunBiomedisa(self, caller=None, event=None) -> None:
        if self._parameterNode.inputVolume and self._parameterNode.inputLabels:
            self.ui.biomedisaButton.enabled = True
        else:
            self.ui.biomedisaButton.enabled = False

    def _checkCanRunSegmentAnything(self, caller=None, event=None) -> None:
        if self._parameterNode.inputVolume and self._parameterNode.inputLabels:
            self.ui.segmentAnythingButton.enabled = True
        else:
            self.ui.segmentAnythingButton.enabled = False

    def _checkCanDeleteLabel(self, caller=None, event=None) -> None:
        if self._parameterNode.inputLabels:
            self.ui.deleteLabelButton.enabled = True
        else:
            self.ui.deleteLabelButton.enabled = False

    def onBiomedisaButton(self) -> None:
        print(f"input: {self._parameterNode.inputVolume.GetName()}")
        print(f"label: {self._parameterNode.inputLabels.GetName()}")
        print ("nbrw:" + str(self._parameterNode.nbrw))
        print ("sorw:" + str(self._parameterNode.sorw))

        inputNode  =  self._parameterNode.inputVolume
        labelsNode  =  self._parameterNode.inputLabels

        inputImage = inputNode.GetImageData()
        labels = labelsNode.GetImageData()

        outputImage = biomedisa_moduleLogic.runBiomedisa(input=inputImage,
                                                         labels=labels, 
                                                         sorw=self._parameterNode.sorw, 
                                                         nbrw=self._parameterNode.nbrw)

        #Set image in view
        if self._parameterNode.outputLabels is labelsNode:
            labelsNode.SetAndObserveImageData(outputImage)

    def onSegmentAnythingButton(self) -> None:
        print(f"input: {self._parameterNode.inputVolume.GetName()}")
        print(f"label: {self._parameterNode.inputLabels.GetName()}")

        sliceIndex = self.get_slice_index()
        foreground = self.get_foreground_points(sliceIndex)[:3]
        background = self.get_background_points(sliceIndex)[:3]

        print(f"sliceIndex: {sliceIndex}")
        print(f"foreground: {foreground}")
        print(f"background: {background}")

        inputNode  =  self._parameterNode.inputVolume

        mask = self.logic.runSegmentAnythingRed(inputNode, sliceIndex, foreground, background)

        # Apply slice to label
        labelsNode  =  self._parameterNode.inputLabels
        labelImage = labelsNode.GetImageData()
        npLabel=vtkNumpyConverter.vtkToNumpy(labelImage)
        npLabel[:, :, sliceIndex] = mask
        vtlLabel = vtkNumpyConverter.numpyToVTK(npLabel)
        labelsNode.SetAndObserveImageData(vtlLabel)

    def onDeleteLabelButton(self) -> None:
        labelsNode  = self._parameterNode.inputLabels
        sliceIndex = self.get_slice_index()
        print(f"deleting slice {sliceIndex}")

        inputImage = labelsNode.GetImageData()
        npImage = vtkNumpyConverter.vtkToNumpy(inputImage)
        npImage[int(sliceIndex), :, :] = 0
        vtlLabel = vtkNumpyConverter.numpyToVTK(npImage)
        labelsNode.SetAndObserveImageData(vtlLabel)

    def onClearPointsButton(self) -> None:
        sliceIndex = self.get_slice_index()
        self.clear_background_points(sliceIndex)
        self.clear_foreground_points(sliceIndex)

#
# biomedisa_moduleLogic
#


class biomedisa_moduleLogic(ScriptedLoadableModuleLogic):
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
        # TODO:
        #print("DEBUG WARNING: predictor removed for development purposes")
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
        return biomedisa_moduleParameterNode(super().getParameterNode())

    def runBiomedisa(
                input: vtkMRMLScalarVolumeNode, #vtkImageData
                labels: vtkMRMLScalarVolumeNode, #vtkImageData
                sorw: int = 1,
                nbrw: int = 1) -> vtkMRMLScalarVolumeNode: #vtkImageData
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        """
        # convert data
        numpyImage = vtkNumpyConverter.vtkToNumpy(input)
        numpyLabels = vtkNumpyConverter.vtkToNumpy(labels)
    
        from biomedisa.interpolation import smart_interpolation
        # smart interpolation with optional smoothing result
        results = smart_interpolation(numpyImage, numpyLabels, nbrw=nbrw, sorw=sorw)#, smooth=100, platform="opencl_AMD_GPU")
       
        # get results
        regular_result = results['regular']
        #smooth_result = results['smooth']

        #print("DEBUG: Code to remove")
        #from biomedisa.features.biomedisa_helper import save_data
        #save_data('C:\\Users\\matze\\Documents\\Code\\biomedisa\\media\\result.tif', regular_result)



        # convert back
        outputImage = vtkNumpyConverter.numpyToVTK(regular_result)
        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        outputVolume.SetAndObserveImageData(outputImage)

        return outputImage

    def runSegmentAnythingRed(self,
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
            point_coords[i] = [point[1], point[0]] # X and Y in np are different than in Slicer
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
        npImage = vtkNumpyConverter.vtkToNumpy(inputImage)
        slice = npImage[:, :, int(index)] # Get Red dimension

        # Stack the grayscale image into three channels to create an RGB image
        grayscale_image = np.array(slice, dtype=np.uint8)
        rgb_image = np.stack((grayscale_image,)*3, axis=-1)
        self.predictor.set_image(rgb_image)

        print(f"rgb_image.dtype: {rgb_image.dtype}")
        print(f"rgb_image.shape: {rgb_image.shape}")

        masks, _, _  = self.predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=False)
        masks_uint8 = (masks* 100).astype(np.uint8) #100 is the segment number

        return masks_uint8[0] # 0 = Take the best one


#
# biomedisa_moduleTest
#


class biomedisa_moduleTest(ScriptedLoadableModuleTest):
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
        self.test_biomedisa_module1()

    def test_biomedisa_module1(self):
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
        inputVolume = SampleData.downloadSample("biomedisa_module1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = biomedisa_moduleLogic()

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
