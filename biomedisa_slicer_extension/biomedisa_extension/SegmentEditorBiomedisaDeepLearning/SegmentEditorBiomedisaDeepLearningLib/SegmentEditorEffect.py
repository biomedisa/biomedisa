import os
import qt, ctk, slicer
from SegmentEditorEffects import *
from Logic.BiomedisaDeepLearningLogic import BiomedisaDeepLearningLogic

# Source: https://github.com/lassoan/SlicerSegmentEditorExtraEffects
class SegmentEditorEffect(AbstractScriptedSegmentEditorEffect):
  """This effect uses the Biomedisa algorithm to segment large 3D volumetric images"""

  def __init__(self, scriptedEffect):
    scriptedEffect.name = 'Biomedisa Deep Learning'
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
    self.stride_size.value = 64, #TODO: dafault to 32
    collapsibleLayout.addRow("Stride size:", self.stride_size)
    
    # Buttons
    self.runButton = qt.QPushButton("Run")
    self.runButton.objectName = self.__class__.__name__ + 'Run'
    self.runButton.setToolTip("Run the biomedisa algorithm and generate segment data")
    self.scriptedEffect.addOptionsWidget(self.runButton)

    self.fileLayout = qt.QHBoxLayout()
    self.cancelButton = qt.QPushButton("Cancel")
    self.cancelButton.objectName = self.__class__.__name__ + 'Cancel'
    self.cancelButton.setToolTip("Clear preview and cancel")
    self.cancelButton.setEnabled(False)
    self.fileLayout.addWidget(self.cancelButton)

    self.selectModelButton = qt.QPushButton("Apply")
    self.selectModelButton.objectName = self.__class__.__name__ + 'Apply'
    self.selectModelButton.setToolTip("Run the biomedisa algorithm and generate segment data")
    self.selectModelButton.setEnabled(False)
    self.fileLayout.addWidget(self.selectModelButton)
    self.scriptedEffect.addOptionsWidget(self.fileLayout)

    self.runButton.connect('clicked()', self.onRun)
    self.cancelButton.connect('clicked()', self.onCancel)
    self.selectModelButton.connect('clicked()', self.onApply)

  def updateApplyButtonState(self):
    if self.previewSegmentationNode is None:
        self.selectModelButton.setEnabled(False)
        self.cancelButton.setEnabled(False)
    else:
        self.selectModelButton.setEnabled(True)
        self.cancelButton.setEnabled(True)

  def removePreviewNode(self):
    if self.previewSegmentationNode:
      slicer.mrmlScene.RemoveNode(self.previewSegmentationNode)
      self.previewSegmentationNode = None
    self.originalSegmentationNode = None
    self.updateApplyButtonState()

  def onSelectModelButton(self):
    fileFilter = "HDF5 Files (*.h5);;All Files (*)"
    fileName = qt.QFileDialog.getOpenFileName(self.selectModelButton, "Select model file", "", fileFilter)
    if fileName:
        self.pathToModel.text = fileName

  def onCancel(self):
    # delete preview segmentation node
    self.runButton.setEnabled(True)
    self.removePreviewNode()

  def onApply(self):
    # move result form preview nod to main node and delete preview segmentation node
    self.originalSegmentationNode.GetSegmentation().DeepCopy(self.previewSegmentationNode.GetSegmentation())
    self.runButton.setEnabled(True)
    self.removePreviewNode()

  def onRun(self):
    # This can be a long operation - indicate it to the user
    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
    try:
      slicer.util.showStatusMessage('Running Biomedisa...', 2000)
      self.runButton.setEnabled(False)
      self.scriptedEffect.saveStateForUndo()
      self.runDeepLearning()
      self.updateApplyButtonState()
      slicer.util.showStatusMessage('Biomedisa finished', 2000)
    except:
      self.runButton.setEnabled(True)
    finally:
      qt.QApplication.restoreOverrideCursor()

  def runDeepLearning(self):
    self.originalSegmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
    self.previewSegmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    self.previewSegmentationNode.SetName("Segmentation preview")
    self.previewSegmentationNode.GetSegmentation().DeepCopy(self.originalSegmentationNode.GetSegmentation())

    displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationDisplayNode")
    displayNode.SetVisibility3D(True)
    displayNode.SetVisibility2DFill(True)
    displayNode.SetVisibility2DOutline(True)
    self.previewSegmentationNode.SetAndObserveDisplayNodeID(displayNode.GetID())

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
      segmentID = segmentation.GetNthSegmentID(int(label) - 1)
      #TODO: create new segments if they don't exist
      slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
        binaryLabelmap, 
        self.previewSegmentationNode, 
        segmentID, 
        slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE, 
        binaryLabelmap.GetExtent())