import os
import qt, ctk, slicer
from SegmentEditorEffects import *
from Logic.BiomedisaLogic import BiomedisaLogic
from Logic.BiomedisaParameter import BiomedisaParameter
from SegmentEditorCommon.AbstractBiomedisaSegmentEditorEffect import AbstractBiomedisaSegmentEditorEffect

# Source: https://github.com/lassoan/SlicerSegmentEditorExtraEffects
class SegmentEditorEffect(AbstractBiomedisaSegmentEditorEffect):
  """This effect uses the Biomedisa algorithm to segment large 3D volumetric images"""

  def __init__(self, scriptedEffect):
    scriptedEffect.name = 'Smart Interpolation'
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
    iconPath = os.path.join(os.path.dirname(__file__), 'SegmentEditorEffect.png')
    if os.path.exists(iconPath):
      return qt.QIcon(iconPath)
    return qt.QIcon()

  def helpText(self):
    return """<html>
      Biomedisa's smart interpolation of sparsely pre-segmented slices<br> creates complete segmentation by considering the underlying image data. Instructions:</p>
      <ul>
        <li>Create complete pre-segmentation on at least one axial slice (red).</li>
        <li>If you label in different orientations from axial, always leave at least one empty slice between pre-segmented slices.</li>
        <li>All visible segments will be interpolated, not just the selected segment.</li>
        <li><b>All axes:</b> enables pre-segmentation in all orientations.</li>
        <li><b>Denoise:</b> smooths/denoises image data before processing.</li>
        <li><b>Number:</b> number of random walks starting at each pre-segmented pixel.</li>
        <li><b>Steps:</b> steps of each random walk.</li>
        <li><b>Ignore:</b> ignores specific label(s), e.g., 2,5,6.</li>
        <li><b>Only:</b> segments only specific label(s), e.g., 1,3,5.</li>
        <li><b>Platform:</b> specifies which platform is used for calculation.</li>
      </ul>
      <p>
        It is a quasi-interpolation because the pre-segmented slices will be adjusted, and segmentation goes beyond the last pre-segmented slices. The effect uses a <a href="https://doi.org/10.1117/12.2216202">diffusion method</a>. For more information, visit the <a href="https://biomedisa.info/">project page</a>.
      </p>
      <p><u>Contributors:</u> <i>Matthias Fabian, Philipp Lösel</i></p>
      <p>
    </html>"""

  def createCursor(self, widget):
    return slicer.util.mainWindow().cursor

  def setupOptionsFrame(self):
    collapsibleButton = ctk.ctkCollapsibleButton()
    collapsibleButton.text = "Advanced"
    collapsibleButton.setChecked(False)
    self.scriptedEffect.addOptionsWidget(collapsibleButton)

    collapsibleLayout = qt.QFormLayout(collapsibleButton)

    self.allaxis = qt.QCheckBox()
    self.allaxis.toolTip = 'If pre-segmentation is not exlusively in xy-plane'
    collapsibleLayout.addRow("All axes:", self.allaxis)

    self.denoise = qt.QCheckBox()
    self.denoise.toolTip = 'Smooth/denoise image data before processing'
    collapsibleLayout.addRow("Denoise:", self.denoise)

    self.nbrw = qt.QSpinBox()
    self.nbrw.toolTip = 'Number of random walks starting at each pre-segmented pixel'
    self.nbrw.minimum = 1
    self.nbrw.maximum = 1000
    self.nbrw.value = 10
    collapsibleLayout.addRow("Number:", self.nbrw)
    
    self.sorw = qt.QSpinBox()
    self.sorw.toolTip = 'Steps of a random walk'
    self.sorw.minimum = 1
    self.sorw.maximum = 1000000
    self.sorw.value = 4000
    collapsibleLayout.addRow("Steps:", self.sorw)

    self.ignore = qt.QLineEdit()
    self.ignore.text = 'none'
    self.ignore.toolTip = 'Ignore specific label(s), e.g. 2,5,6'
    collapsibleLayout.addRow("Ignore:", self.ignore)

    self.only = qt.QLineEdit()
    self.only.text = 'all'
    self.only.toolTip = 'Segment only specific label(s), e.g. 1,3,5'
    collapsibleLayout.addRow("Only:", self.only)

    self.platform = qt.QLineEdit()
    self.platform.text = self.getPlatform()
    self.platform.toolTip = 'One of "cuda", "opencl_NVIDIA_GPU", "opencl_Intel_CPU"'
    collapsibleLayout.addRow("Platform:", self.platform)
    
    AbstractBiomedisaSegmentEditorEffect.setupOptionsFrame(self)

  def getPlatform(self) -> str:
    class Biomedisa:
      def __init__(self, platform=None, success=True, available_devices=0):
        self.platform = platform
        self.success = success
        self.available_devices = available_devices

    from biomedisa.features.biomedisa_helper import _get_platform
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

  def getLabeledSlices(self):
    sourceImageData = self.scriptedEffect.sourceVolumeImageData()
    segmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
    segmentation = segmentationNode.GetSegmentation()
    segmentID = self.scriptedEffect.parameterSetNode().GetSelectedSegmentID()
    segment = segmentation.GetSegment(segmentID)
    binaryLabelmap = segment.GetRepresentation(slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName())
    return BiomedisaLogic.getLabeledSlices(input=sourceImageData, labels=binaryLabelmap)

  def runAlgorithm(self):
    self.createPreviewNode()

    segmentation = self.previewSegmentationNode.GetSegmentation()
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
        self.previewSegmentationNode, 
        segmentID, 
        slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE, 
        binaryLabelmap.GetExtent())
