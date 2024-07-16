import os
from sympy import false
import vtk, qt, ctk, slicer
import logging
from SegmentEditorEffects import *
from Logic.BiomedisaLogic import BiomedisaLogic
from Logic.BiomedisaParameter import BiomedisaParameter
from biomedisa.features.biomedisa_helper import _get_platform

# Source: https://github.com/lassoan/SlicerSegmentEditorExtraEffects
class SegmentEditorEffect(AbstractScriptedSegmentEditorEffect):
  """This effect uses the Biomedisa algorithm to segment large 3D volumetric images"""

  def __init__(self, scriptedEffect):
    scriptedEffect.name = 'Biomedisa'
    scriptedEffect.perSegment = True # this effect operates on a single selected segment
    AbstractScriptedSegmentEditorEffect.__init__(self, scriptedEffect)

  def clone(self):
    # It should not be necessary to modify this method
    import qSlicerSegmentationsEditorEffectsPythonQt as effects
    clonedEffect = effects.qSlicerSegmentEditorScriptedEffect(None)
    clonedEffect.setPythonSource(__file__.replace('\\','/'))
    return clonedEffect

  def icon(self):
    # It should not be necessary to modify this method
    iconPath = os.path.join(os.path.dirname(__file__), 'SegmentEditorEffect.png')
    if os.path.exists(iconPath):
      return qt.QIcon(iconPath)
    return qt.QIcon()

  def helpText(self):
    return """<html>Biomedisa is a free and easy-to-use open-source application for segmenting large volumetric images such as CT and MRI scans,
developed at The Australian National University CTLab. Biomedisa's smart interpolation of sparsely pre-segmented slices
enables accurate semi-automated segmentation by considering the complete underlying image data. 
For more information visit the <a href="https://biomedisa.info/">project page</a>.
<p></html>"""

  def setupOptionsFrame(self):
    collapsibleButton = ctk.ctkCollapsibleButton()
    collapsibleButton.text = "Advanced"
    collapsibleButton.setChecked(False)
    self.scriptedEffect.addOptionsWidget(collapsibleButton)

    collapsibleLayout = qt.QFormLayout(collapsibleButton)

    self.allaxis = qt.QCheckBox()
    self.allaxis.toolTip = 'If pre-segmentation is not exlusively in xy-plane'
    collapsibleLayout.addRow("allaxis:", self.allaxis)

    self.denoise = qt.QCheckBox()
    self.denoise.toolTip = 'Smooth/denoise image data before processing'
    collapsibleLayout.addRow("denoise:", self.denoise)

    self.nbrw = qt.QSpinBox()
    self.nbrw.toolTip = 'Number of random walks starting at each pre-segmented pixel'
    self.nbrw.minimum = 1
    self.nbrw.maximum = 1000
    self.nbrw.value = 10
    collapsibleLayout.addRow("nbrw:", self.nbrw)
    
    self.sorw = qt.QSpinBox()
    self.sorw.toolTip = 'Steps of a random walk'
    self.sorw.minimum = 1
    self.sorw.maximum = 1000000
    self.sorw.value = 4000
    collapsibleLayout.addRow("sorw:", self.sorw)

    self.ignore = qt.QLineEdit()
    self.ignore.text = 'none'
    self.ignore.toolTip = 'Ignore specific label(s), e.g. 2,5,6'
    collapsibleLayout.addRow("ignore:", self.ignore)

    self.only = qt.QLineEdit()
    self.only.text = 'all'
    self.only.toolTip = 'Segment only specific label(s), e.g. 1,3,5'
    collapsibleLayout.addRow("only:", self.only)

    self.platform = qt.QLineEdit()
    self.platform.text = self.getPlatform()
    self.platform.toolTip = 'One of "cuda", "opencl_NVIDIA_GPU", "opencl_Intel_CPU"'
    collapsibleLayout.addRow("platform:", self.platform)

    self.runButton = qt.QPushButton("Run")
    self.runButton.objectName = self.__class__.__name__ + 'Run'
    self.runButton.setToolTip("Run the biomedisa algorithm and generate segment data")

    self.cancelButton = qt.QPushButton("Cancel")
    self.cancelButton.objectName = self.__class__.__name__ + 'Cancel'
    self.cancelButton.setToolTip("Clear preview and cancel")

    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.objectName = self.__class__.__name__ + 'Apply'
    self.applyButton.setToolTip("Run the biomedisa algorithm and generate segment data")

    self.scriptedEffect.addOptionsWidget(self.runButton)

    self.runButton.connect('clicked()', self.onRun)

  def getPlatform(self) -> str:
    class Biomedisa:
      def __init__(self, platform=None, success=True, available_devices=0):
        self.platform = platform
        self.success = success
        self.available_devices = available_devices

    result = Biomedisa(platform=None, success=True, available_devices=0)
    _get_platform(result)
    if(result.success):
      return result.platform
    return 'None'

  def getBiomedisaParameter(self) -> BiomedisaParameter:
    parameter = BiomedisaParameter()
    parameter.allaxis = self.allaxis.isChecked()
    parameter.denoise = self.denoise.isChecked()
    parameter.nbrw = self.nbrw.value
    parameter.sorw = self.sorw.value
    parameter.ignore = self.ignore.text
    parameter.only = self.only.text
    parameter.platform = self.platform.text
    return parameter
    
  def createCursor(self, widget):
    # TODO: Change cursor to make it obvious you are about to do something
    return slicer.util.mainWindow().cursor

  def onRun(self):
    # This can be a long operation - indicate it to the user
    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
    try:
      slicer.util.showStatusMessage('Running Biomedisa...', 2000)
      self.scriptedEffect.saveStateForUndo()
      self.biomedisa()
      slicer.util.showStatusMessage('Biomedisa finished', 2000)

    finally:
      qt.QApplication.restoreOverrideCursor()

  def biomedisa(self):
    segmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
    segmentation = segmentationNode.GetSegmentation()
    segmentID = self.scriptedEffect.parameterSetNode().GetSelectedSegmentID()
    segment = segmentation.GetSegment(segmentID)

    # Get modifier labelmap
    binaryLabelmap = segment.GetRepresentation(slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName())
    # Get source volume image data
    sourceImageData = self.scriptedEffect.sourceVolumeImageData()

    parameter = self.getBiomedisaParameter()
    # Run the algorithm
    resultLabelMaps = BiomedisaLogic.runBiomedisa(input=sourceImageData, labels=binaryLabelmap, parameter=parameter)
    
    for label, binaryLabelmap in resultLabelMaps:
      # Get segment ID from label index. This is 0 based even though first the voxel value is 1.
      segmentID = segmentation.GetNthSegmentID(int(label) - 1)
      slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
        binaryLabelmap, 
        segmentationNode, 
        segmentID, 
        slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE, 
        binaryLabelmap.GetExtent())