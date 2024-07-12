import os
import vtk, qt, slicer
import logging
from SegmentEditorEffects import *
from Logic.BiomedisaLogic import BiomedisaLogic

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
    
    self.allaxis = qt.QCheckBox()
    #self.allaxis.value = False
    self.scriptedEffect.addLabeledOptionsWidget("allaxis:", self.allaxis)

    self.nbrw = qt.QSpinBox()
    self.nbrw.minimum = 1
    self.nbrw.maximum = 1000
    self.nbrw.value = 10
    self.scriptedEffect.addLabeledOptionsWidget("nbrw:", self.nbrw)
    
    self.sorw = qt.QSpinBox()
    self.sorw.minimum = 1
    self.sorw.maximum = 1000000
    self.sorw.value = 4000
    self.scriptedEffect.addLabeledOptionsWidget("sorw:", self.sorw)


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
    # Run the algorithm
    resultLabelMaps = BiomedisaLogic.runBiomedisa(input=sourceImageData, labels=binaryLabelmap)
    
    for label, binaryLabelmap in resultLabelMaps:
      # Get segment ID from label index. This is 0 based even though first the voxel value is 1.
      segmentID = segmentation.GetNthSegmentID(int(label) - 1)
      slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
        binaryLabelmap, 
        segmentationNode, 
        segmentID, 
        slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE, 
        binaryLabelmap.GetExtent())