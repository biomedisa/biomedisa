import os
import qt, ctk, slicer
from SegmentEditorEffects import *
from Logic.BiomedisaTrainingLogic import BiomedisaTrainingLogic
from SegmentEditorCommon.AbstractBiomedisaSegmentEditorEffect import AbstractBiomedisaSegmentEditorEffect

# Source: https://github.com/lassoan/SlicerSegmentEditorExtraEffects
class SegmentEditorEffect(AbstractBiomedisaSegmentEditorEffect):
  """This effect uses the Biomedisa algorithm to segment large 3D volumetric images"""

  def __init__(self, scriptedEffect):
    scriptedEffect.name = 'Biomedisa deep learning training'
    scriptedEffect.perSegment = False
    scriptedEffect.requireSegments = True
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
      <p><u>Contributors:</u> <i>Matthias Fabian, Philipp Lösel<i></p>
      <p></html>"""

  def createCursor(self, widget):
    return slicer.util.mainWindow().cursor
  
  def setupOptionsFrame(self):
    # Network file
    self.pathToModel = qt.QLineEdit()
    #TODO: remove local development path
    #self.pathToModel.text = r"C:\Users\matze\Documents\Code\biomedisa\media\heart\heart.h5" 
    self.pathToModel.toolTip = 'Path of the model file'
    self.pathToModel.textChanged.connect(self.onPathToModelTextChanged)

    self.selectModelButton = qt.QPushButton("...")
    self.selectModelButton.setToolTip("Create a model file")
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
    self.stride_size.value = 64 #TODO: Back to default 32
    collapsibleLayout.addRow("Stride size:", self.stride_size)
    
    self.epochs = qt.QSpinBox()
    self.epochs.toolTip = 'Number of epochs trained'
    self.epochs.minimum = 1
    self.epochs.maximum = 10000
    self.epochs.value = 1 #TODO: Back to default 100
    collapsibleLayout.addRow("Epochs:", self.epochs)
    
    self.x_scale = qt.QSpinBox()
    self.x_scale.toolTip = 'Images and labels are scaled at x-axis to this size before training.'
    self.x_scale.minimum = 1
    self.x_scale.maximum = 4096
    self.x_scale.value = 128 #TODO: Back to default 256
    collapsibleLayout.addRow("X scale:", self.x_scale)
    
    self.y_scale = qt.QSpinBox()
    self.y_scale.toolTip = 'Images and labels are scaled at y-axis to this size before training.'
    self.y_scale.minimum = 1
    self.y_scale.maximum = 4096
    self.y_scale.value = 128 #TODO: Back to default 256
    collapsibleLayout.addRow("Y scale:", self.y_scale)

    self.z_scale = qt.QSpinBox()
    self.z_scale.toolTip = 'Images and labels are scaled at z-axis to this size before training.'
    self.z_scale.minimum = 1
    self.z_scale.maximum = 4096
    self.z_scale.value = 128 #TODO: Back to default 256
    collapsibleLayout.addRow("Z scale:", self.z_scale)

    self.runButton = qt.QPushButton("Train")
    self.runButton.objectName = self.__class__.__name__ + 'Run'
    self.runButton.setToolTip("Run the biomedisa algorithm and generate segment data")
    self.runButton.setEnabled(False)
    self.runButton.connect('clicked()', self.onRun)
    self.scriptedEffect.addOptionsWidget(self.runButton)

  def onPathToModelTextChanged(self):
    # Check if the path is to an existing file
    if os.path.isfile(self.pathToModel.text):
        self.runButton.setEnabled(True)
    else:
        self.runButton.setEnabled(False)
    self.evaluateTrainButton()

  def evaluateTrainButton(self):
    # Evaluate train button enabled
    if self.pathToModel.text.lower().endswith(".h5") and self.hasSegments():
        self.runButton.setEnabled(True)
    else:
        self.runButton.setEnabled(False)

  def hasSegments(self):
    segmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
    segmentation = segmentationNode.GetSegmentation()
    if(not segmentation):
      return False
    if segmentation.GetNumberOfSegments() > 0:
      return True
    return False

  def onSelectModelButton(self):
    fileFilter = "HDF5 Files (*.h5);;All Files (*)"
    fileName = qt.QFileDialog.getSaveFileName(self.selectModelButton, "Create model file", "", fileFilter)
    if fileName:
        self.pathToModel.text = fileName
  
  def runAlgorithm(self):
    segmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
    segmentation = segmentationNode.GetSegmentation()
    segmentID = self.scriptedEffect.parameterSetNode().GetSelectedSegmentID()
    segment = segmentation.GetSegment(segmentID)
    binaryLabelmap = segment.GetRepresentation(slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName())

    sourceImageData = self.scriptedEffect.sourceVolumeImageData()

    # TODO: make sure not to train potentionally faulty data. Maybe have an extra confirm button.

    BiomedisaTrainingLogic.trainDeepLearning(
      sourceImageData, 
      binaryLabelmap, 
      str(self.pathToModel.text),
      stride_size=int(self.stride_size.value),
      epochs=int(self.epochs.value),
      x_scale=int(self.x_scale.value),
      y_scale=int(self.y_scale.value),
      z_scale=int(self.z_scale.value))