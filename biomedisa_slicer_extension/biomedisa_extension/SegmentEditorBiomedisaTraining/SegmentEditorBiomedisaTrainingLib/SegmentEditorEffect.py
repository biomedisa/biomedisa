import os
import qt, ctk, slicer
from SegmentEditorEffects import *
from Logic.BiomedisaTrainingLogic import BiomedisaTrainingLogic
from SegmentEditorCommon.AbstractBiomedisaSegmentEditorEffect import AbstractBiomedisaSegmentEditorEffect

# Source: https://github.com/lassoan/SlicerSegmentEditorExtraEffects
class SegmentEditorEffect(AbstractBiomedisaSegmentEditorEffect):
  """This effect uses the Biomedisa algorithm to segment large 3D volumetric images"""

  def __init__(self, scriptedEffect):
    scriptedEffect.name = 'Biomedisa Training'
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
    return """<html>Deep neural network training for automatic segmentation<br>. Instructions:</p>
      <ul>
        <li>Requires a complete 3D segmentation.</li>
        <li><b>Balance:</b> balance foreground and background training patches.</li>
        <li><b>Swap axes:</b> randomly swaps two axes during training.</li>
        <li><b>Flip x/y/z:</b> randomly flip x/y/z-axis during training.</li>
        <li><b>Scaling:</b> resizes image and label data to dimensions below.</li>
        <li><b>X/Y/Z scale:</b> scales x/y/z-axis of images and labels to this size before training.</li>
        <li><b>Stride size:</b> stride size for extracting patches.</li>
        <li><b>Epochs:</b> number of epochs trained.</li>
        <li><b>Validation split:</b> percentage of data used for training.</li>
      </ul>
      <p>
        The effect uses <a href="https://doi.org/10.1371/journal.pcbi.1011529">deep learning</a>. For more information, visit the <a href="https://biomedisa.info/">project page</a>.
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

    self.balance = qt.QCheckBox()
    self.balance.toolTip = 'Balance foreground and background training patches'
    collapsibleLayout.addRow("Balance:", self.balance)

    self.swapaxes = qt.QCheckBox()
    self.swapaxes.toolTip = 'Randomly swap two axes during training'
    collapsibleLayout.addRow("Swap axes:", self.swapaxes)

    self.flip_x = qt.QCheckBox("x")
    self.flip_x.toolTip = 'Randomly flip x-axis during training'
    self.flip_y = qt.QCheckBox("y")
    self.flip_y.toolTip = 'Randomly flip y-axis during training'
    self.flip_z = qt.QCheckBox("z")
    self.flip_z.toolTip = 'Randomly flip z-axis during training'
    self.flip_layout = qt.QHBoxLayout()
    self.flip_layout.addWidget(self.flip_x)
    self.flip_layout.addStretch()
    self.flip_layout.addWidget(self.flip_y)
    self.flip_layout.addStretch()
    self.flip_layout.addWidget(self.flip_z)
    self.flip_layout.addStretch()
    collapsibleLayout.addRow("Flip:", self.flip_layout)


    self.scaling = qt.QCheckBox()
    self.scaling.toolTip = 'Resize image and label data to dimensions below'
    collapsibleLayout.addRow("Scaling:", self.scaling)

    self.x_scale = qt.QSpinBox()
    self.x_scale.toolTip = 'Images and labels are scaled at x-axis to this size before training.'
    self.x_scale.minimum = 1
    self.x_scale.maximum = 4096
    self.x_scale.value = 256

    self.y_scale = qt.QSpinBox()
    self.y_scale.toolTip = 'Images and labels are scaled at y-axis to this size before training.'
    self.y_scale.minimum = 1
    self.y_scale.maximum = 4096
    self.y_scale.value = 256

    self.z_scale = qt.QSpinBox()
    self.z_scale.toolTip = 'Images and labels are scaled at z-axis to this size before training.'
    self.z_scale.minimum = 1
    self.z_scale.maximum = 4096
    self.z_scale.value = 256

    self.scale_layout = qt.QHBoxLayout()
    self.scale_layout.addWidget(qt.QLabel("X:"))
    self.scale_layout.addWidget(self.x_scale)
    self.scale_layout.addStretch()
    self.scale_layout.addWidget(qt.QLabel("Y:"))
    self.scale_layout.addWidget(self.y_scale)
    self.scale_layout.addStretch()
    self.scale_layout.addWidget(qt.QLabel("Z:"))
    self.scale_layout.addWidget(self.z_scale)
    collapsibleLayout.addRow("Scale:", self.scale_layout)

    self.stride_size = qt.QSpinBox()
    self.stride_size.toolTip = 'Stride size for patches'
    self.stride_size.minimum = 1
    self.stride_size.maximum = 64
    self.stride_size.value = 32
    collapsibleLayout.addRow("Stride size:", self.stride_size)

    self.epochs = qt.QSpinBox()
    self.epochs.toolTip = 'Number of epochs trained'
    self.epochs.minimum = 1
    self.epochs.maximum = 10000
    self.epochs.value = 100
    collapsibleLayout.addRow("Epochs:", self.epochs)

    self.validation_split = qt.QDoubleSpinBox()
    self.validation_split.toolTip = 'Percentage of data used for training'
    self.validation_split.setRange(0.0, 1.0)  # Set the range
    self.validation_split.setDecimals(2)        # Set the number of decimals
    self.validation_split.setSingleStep(0.1)
    self.validation_split.value = 0.0
    collapsibleLayout.addRow("Validation split:", self.validation_split)

    self.runButton = qt.QPushButton("Train")
    self.runButton.objectName = self.__class__.__name__ + 'Run'
    self.runButton.setToolTip("Train neural network")
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
      validation_split=float(self.validation_split.value),
      balance=self.balance.isChecked(),
      swapaxes=self.swapaxes.isChecked(),
      flip_x=self.flip_x.isChecked(),
      flip_y=self.flip_y.isChecked(),
      flip_z=self.flip_z.isChecked(),
      scaling=self.scaling.isChecked(),
      x_scale=int(self.x_scale.value),
      y_scale=int(self.y_scale.value),
      z_scale=int(self.z_scale.value))
