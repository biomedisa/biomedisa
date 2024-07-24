import os
import qt, ctk, slicer
from SegmentEditorEffects import *
from Logic.BiomedisaDeepLearningLogic import BiomedisaDeepLearningLogic
from SegmentEditorCommon.AbstractBiomedisaSegmentEditorEffect import AbstractBiomedisaSegmentEditorEffect

# Source: https://github.com/lassoan/SlicerSegmentEditorExtraEffects
class SegmentEditorEffect(AbstractBiomedisaSegmentEditorEffect):
  """This effect uses the Biomedisa algorithm to segment large 3D volumetric images"""

  def __init__(self, scriptedEffect):
    scriptedEffect.name = 'Biomedisa Prediction'
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
    return """<html>
      Biomedisa Prediction creates a complete 3D segmentation.<br> Instructions:</p>
      <ul>
        <li>Select a model file (trained network).</li>
        <li>Click <i>Initialize</i> to compute preview of full segmentation.</li>
        <li>Click <i>Apply</i> to update segmentation with the previewed result.</li>
        <li><b>Batch size:</b> number of patches per batch. If not specified, it will be adjusted to the available GPU memory.</li>
        <li><b>Stride size:</b> specifies the stride for extracting patches. Increase for a faster but less accurate calculation.</li>
      </ul>
      <p>
        Masking settings are ignored. The effect uses <a href="https://doi.org/10.1371/journal.pcbi.1011529">deep learning</a>. For more information, visit the <a href="https://biomedisa.info/">project page</a>.
      </p>
      <p><u>Contributors:</u> <i>Matthias Fabian, Philipp Lösel</i></p>
      <p>
    </html>"""

  def createCursor(self, widget):
    return slicer.util.mainWindow().cursor
  
  def setupOptionsFrame(self):
    # Network file
    self.pathToModel = qt.QLineEdit()
    #TODO: remove local development path
    self.pathToModel.text = r"C:\Users\matze\Documents\Code\biomedisa\media\heart\heart.h5" 
    self.pathToModel.toolTip = 'Path of the model file'
    self.pathToModel.textChanged.connect(self.onPathToModelTextChanged)

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
    self.stride_size.maximum = 64
    self.stride_size.value = 32
    collapsibleLayout.addRow("Stride size:", self.stride_size)

    self.batch_size_active = qt.QCheckBox()
    self.batch_size_active.toolTip = 'If deactivated the number of patches per batch will be adjusted to the available GPU memory.'
    self.batch_size_active.stateChanged.connect(self.onBatchSizeActiveChanged)

    self.batch_size = qt.QSpinBox()
    self.batch_size.toolTip = 'Number of patches per batch.'
    self.batch_size.minimum = 6
    self.batch_size.maximum = 24
    self.batch_size.value = 12
    self.batch_size.setEnabled(False)

    self.batch_size_layout = qt.QHBoxLayout()
    self.batch_size_layout.addWidget(self.batch_size_active)
    self.batch_size_layout.addWidget(self.batch_size)
    self.batch_size_layout.setStretch(0, 0)
    self.batch_size_layout.setStretch(1, 1)
    collapsibleLayout.addRow("Batch size:", self.batch_size_layout)

    AbstractBiomedisaSegmentEditorEffect.setupOptionsFrame(self)

  def onBatchSizeActiveChanged(self, state):
    self.batch_size.setEnabled(state == qt.Qt.Checked)

  def onPathToModelTextChanged(self):
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

    # Get batch size if checked
    if self.batch_size_active.isChecked():
      batch_size = int(self.batch_size.value)
    else:
      batch_size = None
    
    # Run the algorithm
    resultLabelMaps = BiomedisaDeepLearningLogic.predictDeepLearning(
      input=sourceImageData, 
      modelFile=str(self.pathToModel.text), 
      stride_size=int(self.stride_size.value),
      batch_size=batch_size)

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
