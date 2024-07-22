import os
import qt, ctk, slicer
from SegmentEditorEffects import *
from Logic.BiomedisaDeepLearningLogic import BiomedisaDeepLearningLogic
from SegmentEditorCommon.AbstractBiomedisaSegmentEditorEffect import AbstractBiomedisaSegmentEditorEffect

# Source: https://github.com/lassoan/SlicerSegmentEditorExtraEffects
class SegmentEditorEffect(AbstractBiomedisaSegmentEditorEffect):
  """This effect uses the Biomedisa algorithm to segment large 3D volumetric images"""

  def __init__(self, scriptedEffect):
    scriptedEffect.name = 'Biomedisa deep learning'
    scriptedEffect.perSegment = False
    scriptedEffect.requireSegments = False
    AbstractBiomedisaSegmentEditorEffect.__init__(self, scriptedEffect)

  def clone(self):
    # It should not be necessary to modify this method
    import qSlicerSegmentationsEditorEffectsPythonQt as effects
    clonedEffect = effects.qSlicerSegmentEditorScriptedEffect(None)
    clonedEffect.setPythonSource(__file__.replace('\\','/'))
    return clonedEffect

  def icon(self):
    # It should not be necessary to modify this method
    iconPath = os.path.join(os.path.dirname(__file__), 'SegmentEditorEffect.svg')
    if os.path.exists(iconPath):
      return qt.QIcon(iconPath)
    return qt.QIcon()

  def helpText(self):
    return """<html>Biomedisa is a free and easy-to-use open-source application for
      segmenting large volumetric images<br> such as CT and MRI scans, developed at The Australian National University CTLab. Biomedisa's smart interpolation of sparsely pre-segmented slices
      enables accurate semi-automated segmentation by considering the complete underlying image data.</p>
      
      <p>For more information visit the <a href="https://biomedisa.info/">project page</a>.</p>
      <p>Instructions:
      <ul>
        <li>Create segments on at least one axial layer</li>
        <li>Run the algorithm</li>
        <li>???</li>
        <li>Profit</li>
      </ul>
      </p>
      <p><u>Contributors:</u> <i>Matthias Fabian, Dr. Philipp Lösel<i></p>
      <p></html>"""

  def createCursor(self, widget):
    return slicer.util.mainWindow().cursor
  
  def setupOptionsFrame(self):
    # Network file
    self.pathToModel = qt.QLineEdit()
    #TODO: remove local development path
    self.pathToModel.text = r"C:\Users\matze\Documents\Code\biomedisa\media\heart\heart.h5" 
    self.pathToModel.toolTip = 'Path of the model file'
    self.pathToModel.textChanged.connect(self.validatePath)

    self.selectModelButton = qt.QPushButton("...")
    self.selectModelButton.setToolTip("Select a model file")
    self.selectModelButton.connect('clicked()', self.onSelectModelButton)
    
    self.fileLayout = qt.QHBoxLayout()
    self.fileLayout.addWidget(self.pathToModel)
    self.fileLayout.addWidget(self.selectModelButton)
    self.scriptedEffect.addOptionsWidget(self.fileLayout)

    # Advanced menu
    collapsibleButton = ctk.ctkCollapsibleButton()
    collapsibleButton.text = "Advanced"
    collapsibleButton.setChecked(False)
    self.scriptedEffect.addOptionsWidget(collapsibleButton)

    collapsibleLayout = qt.QFormLayout(collapsibleButton)

    self.stride_size = qt.QSpinBox()
    self.stride_size.toolTip = 'Stride size for patches'
    self.stride_size.minimum = 1
    self.stride_size.maximum = 65
    self.stride_size.value = 32
    collapsibleLayout.addRow("Stride size:", self.stride_size)

    AbstractBiomedisaSegmentEditorEffect.setupOptionsFrame(self)

  def validatePath(self):
    # Check if the path is to an existing file
    if os.path.isfile(self.pathToModel.text):
        self.runButton.setEnabled(True)
    else:
        self.runButton.setEnabled(False)

  def onSelectModelButton(self):
    fileFilter = "HDF5 Files (*.h5);;All Files (*)"
    fileName = qt.QFileDialog.getOpenFileName(self.selectModelButton, "Select model file", "", fileFilter)
    if fileName:
        self.pathToModel.text = fileName

  def runAlgorithm(self):
    self.createPreviewNode()

    # Get source volume image data
    sourceImageData = self.scriptedEffect.sourceVolumeImageData()

    # Run the algorithm
    resultLabelMaps = BiomedisaDeepLearningLogic.runDeepLearning(
      input=sourceImageData, 
      modelFile=str(self.pathToModel.text), 
      stride_size=int(self.stride_size.value))

    # Show the result in slicer
    segmentation = self.previewSegmentationNode.GetSegmentation()
    for label, binaryLabelmap in resultLabelMaps:
      # Get segment ID from label index. This is 0 based even though first the voxel value is 1.
      segmentID = segmentation.AddEmptySegment()
      slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
        binaryLabelmap, 
        self.previewSegmentationNode, 
        segmentID, 
        slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE, 
        binaryLabelmap.GetExtent())