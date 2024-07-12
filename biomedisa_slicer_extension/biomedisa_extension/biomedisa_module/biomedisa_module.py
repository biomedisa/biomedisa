import math
import os
from typing import Annotated, Optional
import time
from unittest import result

import numpy as np
import vtk
from vtk import vtkCommand
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
from slicer import vtkMRMLSegmentationNode
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
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Matthias Fabian"]
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
<h1>Biomedisa</h1>
Biomedisa is a free and easy-to-use open-source application for segmenting large volumetric images such as CT and MRI scans,
developed at The Australian National University CTLab. Biomedisa's smart interpolation of sparsely pre-segmented slices
enables accurate semi-automated segmentation by considering the complete underlying image data. 
For more information visit the <a href="https://biomedisa.info/">project page</a>.
                                 
<h3>How to use:</h3>
<ol>
    <li>Add segment points (blue) by clicking in the red image.</li>
    <li>Add non-segment points (green) by holding down the Alt key and clicking in the red image.</li>
    <li>Run the segment anything algorithm to create a label mask for the current layer.</li>
    <li>Run the biomedisa algorithm to create a label mask for the entire 3D image.</li>
</ol>
""")
        self.parent.acknowledgementText = _("""
                                            This extension was cooked up by a jobless C#.NET dev diving into Python.
                                            Apologies for the spaghetti code.
                                            Shout-out to Germany for funding this project with unemployment benefits!
""")

#
# biomedisa_moduleParameterNode
#


# Example of connecting a custom callback to the module loaded event

@parameterNodeWrapper
class biomedisa_moduleParameterNode:
    """
    The parameters needed by module.

    """

    inputVolume: vtkMRMLScalarVolumeNode
    inputLabels: vtkMRMLLabelMapVolumeNode
    segmentation: vtkMRMLSegmentationNode

    #biomedisa
    allaxis: bool = False
    nbrw: int = 10
    sorw: int = 4000

    #segment anything
    segmentAnythingActive: bool = False
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 16s loading, 36s image setting
    #modelCheckpointFile: str = os.path.join(script_dir, "Resources", "sam_vit_h_4b8939.pth")
    #modelType:str = "vit_h"
    #modelCheckpointFile: str = os.path.join(script_dir, "Resources", "sam_vit_l_0b3195.pth")
    #modelType:str = "vit_l"
    # 1s loading, 8s image setting
    modelCheckpointFile: str = os.path.join(script_dir, "Resources", "sam_vit_b_01ec64.pth")
    modelType:str = "vit_b"


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

        self.parameterSetNode = None
        self.editor = None
        
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
        #self.ui.labelSelector.setMRMLScene(slicer.mrmlScene)
        
        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = biomedisa_moduleLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.__setupMarkups()

        # Buttons
        self.ui.biomedisaButton.connect("clicked(bool)", self.onBiomedisaButton)
        self.ui.deleteLabelButton.connect("clicked(bool)", self.onDeleteLabelButton)
        self.ui.clearPointsButton.connect("clicked(bool)", self.onClearPointsButton)
        self.ui.trainPredictorButton.connect("clicked(bool)", self.onTrainPredictorButton)

        #
        # Segment editor widget
        #
        import qSlicerSegmentationsModuleWidgetsPythonQt
        self.editor = qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
        self.editor.setMaximumNumberOfUndoStates(10)
        # Set parameter node first so that the automatic selections made when the scene is set are saved
        self.selectParameterNode()
        self.editor.setMRMLScene(slicer.mrmlScene)
        self.layout.insertWidget(1, self.editor)
        # Find the qMRMLSegmentsTableView widget
        self.segmentsTableView = self.editor.findChild(slicer.qMRMLSegmentsTableView, 'SegmentsTableView')

        # Make sure parameter node is initialized (needed for module reload)
        self.__initializeParameterNode()

    def selectParameterNode(self):
        print("selectParameterNode")
        # Select parameter set node if one is found in the scene, and create one otherwise
        segmentEditorSingletonTag = "SegmentEditor"
        segmentEditorNode = slicer.mrmlScene.GetSingletonNode(segmentEditorSingletonTag, "vtkMRMLSegmentEditorNode")
        if segmentEditorNode is None:
            segmentEditorNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            segmentEditorNode.UnRegister(None)
            segmentEditorNode.SetSingletonTag(segmentEditorSingletonTag)
            segmentEditorNode = slicer.mrmlScene.AddNode(segmentEditorNode)
        if self.parameterSetNode == segmentEditorNode:
            # nothing changed
            return
        self.parameterSetNode = segmentEditorNode
        self.editor.setMRMLSegmentEditorNode(self.parameterSetNode)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()
        self.__clearAllPoins()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.__initializeParameterNode()
        # Add observer for left mouse button click
        self.observerId = self.interactor.AddObserver(vtkCommand.LeftButtonPressEvent, self.onLeftClick)
        
        self.fgMkNPAddObserverId = self.foregroundMarkupsNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointRemovedEvent, self.onControlPointsUpdated)
        self.fgMkNPEndObserverId = self.foregroundMarkupsNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent, self.onControlPointsUpdated)
        self.bgMkNPAddObserverId = self.backgroundMarkupsNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointRemovedEvent, self.onControlPointsUpdated)
        self.bgMkNPEndObserverId = self.backgroundMarkupsNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointEndInteractionEvent, self.onControlPointsUpdated)

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.__checkCanRunBiomedisa)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.__checkCanDeleteLabel)
        if(hasattr(self, 'interactor') and hasattr(self, 'observerId')):
            self.interactor.RemoveObserver(self.observerId)
        if(hasattr(self, 'foregroundMarkupsNode')):
            if (hasattr(self, 'fgMkNPAddObserverId')):
                self.foregroundMarkupsNode.RemoveObserver(self.fgMkNPAddObserverId)
            if (hasattr(self, 'fgMkNPEndObserverId')):
                self.foregroundMarkupsNode.RemoveObserver(self.fgMkNPEndObserverId)
        if(hasattr(self, 'backgroundMarkupsNode')):
            if (hasattr(self, 'bgMkNPAddObserverId')):
                self.backgroundMarkupsNode.RemoveObserver(self.bgMkNPAddObserverId)
            if (hasattr(self, 'bgMkNPEndObserverId')):
                self.backgroundMarkupsNode.RemoveObserver(self.bgMkNPEndObserverId)
    
    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.__setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.__initializeParameterNode()

    def getBinaryLabelMap(self):
        segment = self.getSegment()
        binaryLabelmap = segment.GetRepresentation(slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName())
        return binaryLabelmap

    def setBinaryLabelMap(self, label: np.array):
        import vtk.util.numpy_support as vtk_np
        vtkImageData = slicer.vtkOrientedImageData()
        vtkImageData.SetDimensions(label.shape[2], label.shape[1], label.shape[0])
        vtkImageData.SetDirections([[-1,0,0],[0,-1,0],[0,0,1]])
        vtkImageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        vtkArray = vtk_np.numpy_to_vtk(num_array=label.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtkImageData.GetPointData().SetScalars(vtkArray)

        # Update the segment with the modified binary labelmap
        segment = self.getSegment()
        segment.AddRepresentation(slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName(), vtkImageData)
        self.getSegmentationNode().Modified()

    def setBinaryLabelMapSlice(self, mask, segmentLayer = 1):
        # Get the segmentation node and the segment
        segmentationNode = self.getSegmentationNode()
        segment = self.getSegment()
        sliceIndex = self.__getSliceIndex()

        mask = (mask* segmentLayer).astype(np.uint8) #100 is the segment number

        # Get the current binary labelmap representation
        binaryLabelmap = segment.GetRepresentation(slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName())

        # Make a deep copy of the existing binary labelmap to modify
        modifiedLabelmap = slicer.vtkOrientedImageData()
        modifiedLabelmap.DeepCopy(binaryLabelmap)

        # Get the extent of the existing binary labelmap
        extent = modifiedLabelmap.GetExtent()

        # Check if the sliceIndex is within the extent, and extend if necessary
        if sliceIndex < extent[4] or sliceIndex > extent[5]:
            if extent[1] == -1 and extent[3] == -1 and extent[5] == -1:
                newExtent = [0, mask.shape[1]-1, 
                             0, mask.shape[0]-1,
                             sliceIndex, sliceIndex]
            else:
                newExtent = [min(extent[0], mask.shape[1]-1), 
                            max(extent[1], mask.shape[1]-1),
                            min(extent[2], mask.shape[0]-1), 
                            max(extent[3], mask.shape[0]-1),
                            min(extent[4], sliceIndex), 
                            max(extent[5], sliceIndex)]

            # Create a new vtkOrientedImageData with the new extent
            newModifiedLabelmap = slicer.vtkOrientedImageData()
            newModifiedLabelmap.SetExtent(newExtent)
            newModifiedLabelmap.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
            newModifiedLabelmap.SetDirections([[-1,0,0],[0,-1,0],[0,0,1]])

            # Copy data from the original to the new labelmap
            for z in range(extent[4], extent[5] + 1):
                for y in range(extent[2], extent[3] + 1):
                    for x in range(extent[0], extent[1] + 1):
                        newModifiedLabelmap.SetScalarComponentFromFloat(x, y, z, 0, modifiedLabelmap.GetScalarComponentAsFloat(x, y, z, 0))

            modifiedLabelmap = newModifiedLabelmap
            extent = newExtent

        for y in range(mask.shape[0]):
            for x in range(max(extent[0], mask.shape[1])):
                if extent[0] <= x <= extent[1] and extent[2] <= y <= extent[3]:
                    # This will throw exceptions if the coordinates are not in the extent
                    modifiedLabelmap.SetScalarComponentFromFloat(x, y, sliceIndex, 0, mask[y, x])

        # Update the segment with the modified binary labelmap
        segment.AddRepresentation(
            slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName(), modifiedLabelmap)

        # Notify the segmentation node that the binary labelmap has been modified
        segmentationNode.Modified()

    def onLeftClick(self, caller, event):
        """Called when the left mouse button is clicked."""

        if(not self._parameterNode.segmentAnythingActive):
            return

        xy = self.interactor.GetEventPosition()

        sliceValue = self.__getSliceIndex()

        ras = self.slice_view.convertXYZToRAS([xy[0], xy[1], sliceValue])
        ras = [ras[0], ras[1], sliceValue]

        # Check if out of frame and return
        dims = self._parameterNode.inputVolume.GetImageData().GetDimensions()
        if (-ras[0] < 0 or -ras[0] > dims[0] or -ras[1] < 0 or -ras[1] > dims[1]):
            return

        modifiers = QGuiApplication.queryKeyboardModifiers()
        alt_pressed = bool(modifiers & Qt.Modifier.ALT)
        if(alt_pressed):
            self.backgroundMarkupsNode.AddControlPoint(ras[0], ras[1], ras[2])
        else:
            self.foregroundMarkupsNode.AddControlPoint(ras[0], ras[1], ras[2])

        self.__runSegmentAnything()

    def onControlPointsUpdated(self, caller=None, event=None) -> None:
        if(not self._parameterNode.segmentAnythingActive):
            return
        self.__runSegmentAnything()

    def onInpuVolumeChanged(self, node):
        #TODO: connect this to the segmenteditor's volume choice
        if node is not None:
            sliceIndex = self.__getSliceIndex()
            self.logic.setSegmentAnythingImage(node, sliceIndex)

    def onBiomedisaButton(self) -> None:
        inputNode  =  self.getVolumeNode()
        inputImage = inputNode.GetImageData()
        labels = self.getBinaryLabelMap()

        outputImage = biomedisa_moduleLogic.runBiomedisa(input=inputImage,
                                                         labels=labels, 
                                                         allaxis=self._parameterNode.allaxis,
                                                         sorw=self._parameterNode.sorw, 
                                                         nbrw=self._parameterNode.nbrw)
        
        if(outputImage is None):
            return
        
        self.setBinaryLabelMap(outputImage)

    def onDeleteLabelButton(self) -> None:
        labelsNode  = self._parameterNode.inputLabels
        sliceIndex = self.__getSliceIndex()
        print(f"deleting slice {sliceIndex}")

        inputImage = labelsNode.GetImageData()
        npImage = vtkNumpyConverter.vtkToNumpy(inputImage)
        npImage[int(sliceIndex), :, :] = 0
        vtlLabel = vtkNumpyConverter.numpyToVTK(npImage)
        labelsNode.SetAndObserveImageData(vtlLabel)

    def onClearPointsButton(self) -> None:
        sliceIndex = self.__getSliceIndex()
        self.__clearBackgroundPoints(sliceIndex)
        self.__clearForegroundPoints(sliceIndex)

    def onTrainPredictorButton(self) -> None:
        self.logic.setupPredictor()

    def getVolumeNode(self) -> vtkMRMLScalarVolumeNode:
        return self.editor.SourceVolumeNodeComboBox.currentNode()

    def getSegmentationNode(self) -> vtkMRMLSegmentationNode:
        return self.editor.SegmentationNodeComboBox.currentNode()

    def getSegmentation(self):# -> vtkSegmentation:
        return self.getSegmentationNode().GetSegmentation()

    def getSegmentID(self) -> str:
        ids = self.segmentsTableView.selectedSegmentIDs() #returns list[str]
        return ids[0]

    def getSegment(self):# -> vtkSegment:
        segmentation = self.getSegmentation()
        id = self.getSegmentID()
        return segmentation.GetSegment(id)
    
    def __initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.__setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            nodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
            for node in nodes:
                if not isinstance(node, vtkMRMLLabelMapVolumeNode):
                    self._parameterNode.inputVolume = node
                    break

    def __setParameterNode(self, inputParameterNode: Optional[biomedisa_moduleParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.__checkCanRunBiomedisa)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.__checkCanDeleteLabel)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.__checkCanRunBiomedisa)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.__checkCanDeleteLabel)
            self.__checkCanRunBiomedisa()
            self.__checkCanDeleteLabel()

    def __setupMarkups(self):
        self.slice_widget = slicer.app.layoutManager().sliceWidget('Red')
        self.slice_view = self.slice_widget.sliceView()
        self.interactor = self.slice_view.interactorStyle().GetInteractor()

        self.foregroundMarkupsNode = self.__createMarkups(self.slice_widget, "Segment", 0, 0, 1)
        self.backgroundMarkupsNode = self.__createMarkups(self.slice_widget, "Non Segement", 1, 0, 0)

    def __createMarkups(self ,slice_widget, name, r, g, b):
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

    def __getSliceIndex(self )-> int: 
        """Returns the currently selected layer index based on the slider value."""
        sliceController = self.slice_widget.sliceController()
        sliceValue = sliceController.sliceOffsetSlider().value
        return int(sliceValue)
        
    def __getForegroundPoints(self, sliceIndex):
        return self.__getPoints(self.foregroundMarkupsNode, sliceIndex)

    def __getBackgroundPoints(self, sliceIndex):
        return self.__getPoints(self.backgroundMarkupsNode, sliceIndex)

    def __getPoints(self, markupsNode, sliceIndex):
        points = []
        pointsCount = markupsNode.GetNumberOfControlPoints()
        for i in range(pointsCount):
            coords = [0.0, 0.0, 0.0]
            markupsNode.GetNthControlPointPosition(i, coords)
            if(coords[2] == sliceIndex):
                points.append([-round(coords[0]), -round(coords[1]), coords[2], i])
        return points

    def __clearBackgroundPoints(self, sliceIndex):
        self.__clearPoints(self.backgroundMarkupsNode, sliceIndex)

    def __clearForegroundPoints(self, sliceIndex):
        self.__clearPoints(self.foregroundMarkupsNode, sliceIndex)

    def __clearPoints(self, markupsNode, sliceIndex):
        """Removes all control points in current layer."""
        sliceIndex = self.__getSliceIndex()
        points = self.__getPoints(markupsNode, sliceIndex)
        for point in reversed(points):
            markupsNode.RemoveNthControlPoint(point[3])

    def __clearAllPoins(self):
        """Removes all control points."""
        self.foregroundMarkupsNode.RemoveAllControlPoints()
        self.backgroundMarkupsNode.RemoveAllControlPoints()

    def __runSegmentAnything(self) -> None:
        """Collects data in widget, starts the process in logic and displays the result."""

        sliceIndex = self.__getSliceIndex()
        foreground = self.__getForegroundPoints(sliceIndex)
        background = self.__getBackgroundPoints(sliceIndex)
        inputNode  = self.getVolumeNode()

        mask = self.logic.runSegmentAnythingRed(inputNode, sliceIndex, foreground, background)

        #TODO: Segment index needs to come from segment selection in segmentsTableView
        self.setBinaryLabelMapSlice(mask, 1)

    def __checkCanRunBiomedisa(self, caller=None, event=None) -> None:
        self.ui.biomedisaButton.enabled = True
        return
        #TODO: check as below
        if self.getVolumeNode() and self.getSegmentation():
            self.ui.biomedisaButton.enabled = True
        else:
            self.ui.biomedisaButton.enabled = False

    def __checkCanDeleteLabel(self, caller=None, event=None) -> None:
        if self._parameterNode.inputLabels:
            self.ui.deleteLabelButton.enabled = True
        else:
            self.ui.deleteLabelButton.enabled = False


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

        self.parameterNode = self.getParameterNode()
        
        self.predictorSliceIndex = None
        self.predictorVolumeName = None

    def setupPredictor(self):
        """
        Prepares the segment anything predictor.
        Depending on the model file size this will take several seconds. Approx. 1s/200MB.
        """
        startTime = time.time()
        
        if not os.path.exists(self.parameterNode.modelCheckpointFile):
            raise Exception(f"You need to download a model checkpoint and store it at {parameter.modelCheckpointFile}. Check out: https://github.com/facebookresearch/segment-anything")
        print(f"Training predictor with modelType '{self.parameterNode.modelType}' and file '{self.parameterNode.modelCheckpointFile}'.")
        
        sam = sam_model_registry[self.parameterNode.modelType](checkpoint=self.parameterNode.modelCheckpointFile)
        self.predictor = SamPredictor(sam)
        endTime = time.time()
        print(f"Predictor set up in {endTime-startTime:.2f} s")

    def getParameterNode(self):
        return biomedisa_moduleParameterNode(super().getParameterNode())

    def expandLabelToMatchInputImage(labelImageData, inputDimensions):
        import vtk.util.numpy_support as vtk_np
        # Initialize the new VTK image data object with the same dimensions as the input image
        newLabelImageData = vtk.vtkImageData()
        newLabelImageData.SetDimensions(inputDimensions)
        newLabelImageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        #newModifiedLabelmap.SetDirections([[-1,0,0],[0,-1,0],[0,0,1]])

        # Get the bounds and extent of the original label image data
        labelBounds = labelImageData.GetBounds()
        labelExtent = labelImageData.GetExtent()
        
        print(f"Label Bounds: {labelBounds}")
        print(f"Label Extent: {labelExtent}")

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
        print(offsets)
        print(f"labelNumpyArray: {labelNumpyArray.shape}")
        print(f"newLabelNumpyArray: {newLabelNumpyArray.shape}")

        # Iterate over the label data and copy it to the new image data at the correct position
        rz =range(labelExtent[4], labelExtent[5] + 1)
        ry = range(labelExtent[2], labelExtent[3] + 1)
        rx = range(labelExtent[0], labelExtent[1] + 1)
        print(f"rx {rx}")
        print(f"ry {ry}")
        print(f"rz {rz}")
        for z in rz:
            for y in ry:
                for x in rx:
                    zz = int(z - offsets[2])
                    yy = int(y - offsets[1])
                    xx = int(x - offsets[0])
                    #print(f"xzy: {xx},{yy},{zz} -> {x},{y},{z}")
                    v = labelNumpyArray[zz, yy, xx]
                    newLabelNumpyArray[z, y, x] = v

        # Convert the NumPy array back to a VTK array and set it as the scalars of the new VTK image data object
        newLabelVtkArray = vtk_np.numpy_to_vtk(newLabelNumpyArray.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        newLabelImageData.GetPointData().SetScalars(newLabelVtkArray)

        return newLabelImageData

    def runBiomedisa(
                input: vtkMRMLScalarVolumeNode,
                labels: vtkMRMLLabelMapVolumeNode, 
                allaxis: bool,
                sorw: int,
                nbrw: int) -> np.array:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param input: volume to calculate on
        :param labels: mask to use as a base
        :param output: the resulting mask result
        """
        extendedLabel = biomedisa_moduleLogic.expandLabelToMatchInputImage(labels, input.GetDimensions())
        # convert data
        numpyImage = vtkNumpyConverter.vtkToNumpy(input)
        numpyLabels = vtkNumpyConverter.vtkToNumpy(extendedLabel)

        from biomedisa.interpolation import smart_interpolation
        # smart interpolation with optional smoothing result
        results = smart_interpolation(numpyImage, numpyLabels,
                                      allaxis=allaxis,
                                      nbrw=nbrw, sorw=sorw)
        if results is None:
            return None
        
        # get results
        regular_result = results['regular']
        #smooth_result = results['smooth']

        return regular_result

    def setSegmentAnythingImage(self,
                                inputVolume: vtkMRMLScalarVolumeNode,
                                index: int):
        """
        Applies the image to the segment anything predictor.
        Depending on the model file size and image size this will take several seconds. 
        """
        
        if(not self.parameterNode.segmentAnythingActive):
            return
        
        if(not hasattr(self, 'predictor')):
            self.setupPredictor()

        if(not hasattr(self, 'predictor')):
            raise Exception("Predictor is not trained. Make sure you've got a working model checkpoint and type.")

        startTime = time.time()
        inputImage = inputVolume.GetImageData()
        npImage = vtkNumpyConverter.vtkToNumpy(inputImage)
        slice = npImage[int(index), :, :] # Get Red dimension

        # Stack the grayscale image into three channels to create an RGB image
        grayscale_image = np.array(slice, dtype=np.uint8)
        rgb_image = np.stack((grayscale_image,)*3, axis=-1)
        self.predictor.set_image(rgb_image)
        self.predictorSliceIndex = index
        self.predictorVolumeName = inputVolume.GetName()
        endTime = time.time()
        print(f"Predictor image is set in {endTime-startTime:.2f}")

    def isPredictorImageSet(self,
                   inputVolume: vtkMRMLScalarVolumeNode,
                   index: int):
        return self.predictorSliceIndex == index and self.predictorVolumeName == inputVolume.GetName()

    def runSegmentAnythingRed(self,
                              inputVolume: vtkMRMLScalarVolumeNode,
                              index: int,
                              foreground: np.array,
                              background: np.array):
        if(not hasattr(self, 'predictor')):
            self.setupPredictor()

        startTime = time.time()
        length_f = len(foreground)
        length_b = len(background)

        point_coords = np.empty((length_f+length_b, 2), dtype=int)
        point_labels = np.empty((length_f+length_b), dtype=int)

        for i in range(length_f):
            point = foreground[i]
            point_coords[i] = [point[0], point[1]]
            point_labels[i] = 1
        for i in range(length_b):
            point = background[i]
            point_coords[length_f+i] = [point[0], point[1]]
            point_labels[length_f+i] = 0

        if(not self.isPredictorImageSet(inputVolume, index)):
            self.setSegmentAnythingImage(inputVolume, index)

        masks, _, _  = self.predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=False)
        endTime = time.time()
        print(f"Segment anything completed within {endTime-startTime:.2f} s")
        return masks[0] # 0 = Take the best one


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

        self.delayDisplay("Test passed")
