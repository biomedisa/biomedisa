import os, qt, ctk, slicer, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(root_dir)

from biomedisa_extension.SegmentEditorBiomedisaPrediction.Logic.BiomedisaPredictionParameter import BiomedisaPredictionParameter
from biomedisa_extension.SegmentEditorBiomedisaPrediction.Logic.BiomedisaPredictionLogic import BiomedisaPredictionLogic
from biomedisa_extension.SegmentEditorCommon.AbstractBiomedisaSegmentEditorEffect import AbstractBiomedisaSegmentEditorEffect
from biomedisa_extension.SegmentEditorCommon.ListSelectionDialog import ListSelectionDialog
from biomedisa_extension.SegmentEditorCommon.RoiSelectionWidget import ROISelectionWidget

class SegmentEditorEffect(AbstractBiomedisaSegmentEditorEffect):

  def __init__(self, scriptedEffect):
    scriptedEffect.perSegment = False
    scriptedEffect.requireSegments = False
    super().__init__(scriptedEffect, 'Biomedisa Prediction')

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
  
  def setupOptionsFrame(self):
    # Network file
    self.path_to_model = qt.QLineEdit()
    self.path_to_model.toolTip = 'Path of the model file'
    self.path_to_model.setPlaceholderText('Enter the path of the model file here...')
    self.path_to_model.textChanged.connect(self.onPathToModelTextChanged)

    self.selectModelButton = qt.QPushButton("...")
    self.selectModelButton.setFixedWidth(30)
    self.selectModelButton.setToolTip("Select a model file")
    self.selectModelButton.clicked.connect(self.onSelectModelButton)
    
    self.fileLayout = qt.QHBoxLayout()
    self.fileLayout.addWidget(self.path_to_model)
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

    self.roiSelectionWidget = ROISelectionWidget(self.scriptedEffect, "Biomedisa Prediction ROI")
    collapsibleLayout.addRow("Prediction area", self.roiSelectionWidget)

    self.setParameterToGui(BiomedisaPredictionParameter())

    collapsibleLayout.addRow("Parameter:", self.createParameterGui())
    AbstractBiomedisaSegmentEditorEffect.setupOptionsFrame(self)

  def sourceVolumeNodeChanged(self):
    sourceImageData = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode().GetImageData()
    if sourceImageData is None:
      return
    dim = sourceImageData.GetDimensions()

  def onLoadParameter(self):
    parameterList = self.getSavedParameter()
    self.dialog = ListSelectionDialog(parameterList)
    def handleDialogClosed(selected_item):
      if selected_item:
        parameter = self.loadParameter(BiomedisaPredictionParameter, selected_item)
        parameter.name = selected_item
        self.setParameterToGui(parameter)
    self.dialog.dialogClosed.connect(handleDialogClosed)
    self.dialog.show()

  def onRestoreParameter(self):
    parameter = BiomedisaPredictionParameter()
    self.setParameterToGui(parameter)

  def getParameterFromGui(self) -> BiomedisaPredictionParameter:
    roiXYZ = self.roiSelectionWidget.getXYZMinMax()

    parameter = BiomedisaPredictionParameter()
    parameter.path_to_model = self.path_to_model.text
    parameter.stride_size = self.stride_size.value
    parameter.batch_size_active = self.batch_size_active.isChecked()
    parameter.batch_size = self.batch_size.value
    parameter.x_min = roiXYZ[0]
    parameter.x_max = roiXYZ[1]
    parameter.y_min = roiXYZ[2]
    parameter.y_max = roiXYZ[3]
    parameter.z_min = roiXYZ[4]
    parameter.z_max = roiXYZ[5]
    return parameter
  
  def setParameterToGui(self, parameter: BiomedisaPredictionParameter):
    self.path_to_model.text = parameter.path_to_model
    self.stride_size.value = parameter.stride_size
    self.batch_size_active.setChecked(parameter.batch_size_active)
    self.batch_size.value = parameter.batch_size
    self.roiSelectionWidget.setXYZMinMax(parameter.x_min, parameter.x_max,
                                         parameter.y_min, parameter.y_max, 
                                         parameter.z_min, parameter.z_max,
                                         parameter.name)

  def onBatchSizeActiveChanged(self, state):
    self.batch_size.setEnabled(state == qt.Qt.Checked)

  def onPathToModelTextChanged(self):
    self.updateRunButtonState()

  def updateRunButtonState(self):
    if self.running:
        self.runButton.setEnabled(False)
    elif os.path.isfile(self.path_to_model.text):
        self.runButton.setEnabled(True)
    else:
        self.runButton.setEnabled(False)

  def onSelectModelButton(self):
    fileFilter = "HDF5 Files (*.h5);;All Files (*)"
    fileName = qt.QFileDialog.getOpenFileName(self.selectModelButton, "Select model file", "", fileFilter)
    if fileName:
        self.path_to_model.text = fileName

  def runAlgorithm(self):
    self.createPreviewNode()

    # Get source volume image data
    volumeNode = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode()
    sourceImageData = volumeNode.GetImageData()

    # Run the algorithm
    resultLabelMaps = BiomedisaPredictionLogic.predictDeepLearning(
      input=sourceImageData,
      volumeNode=volumeNode,
      parameter=self.getParameterFromGui())

    # Show the result in slicer
    segmentation = self.previewSegmentationNode.GetSegmentation()
    segmentIDs = segmentation.GetSegmentIDs()
    availableLabelValues = {segmentation.GetSegment(segmentID).GetLabelValue(): segmentID for segmentID in segmentIDs}

    for label, binaryLabelmap in resultLabelMaps:
      # Map results to the corresponding segements
      segmentID = availableLabelValues.get(label)
      if not segmentID:
        segmentID = segmentation.AddEmptySegment()
        segmentation.GetSegment(segmentID).SetLabelValue(label)

      segment = segmentation.GetSegment(segmentID)
      segment.SetLabelValue(label)
      slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
        binaryLabelmap, 
        self.previewSegmentationNode, 
        segmentID, 
        slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE, 
        binaryLabelmap.GetExtent())
