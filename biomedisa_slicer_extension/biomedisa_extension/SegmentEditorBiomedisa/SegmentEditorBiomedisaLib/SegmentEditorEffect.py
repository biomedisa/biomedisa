import os, qt, ctk, slicer, sys
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from SegmentEditorCommon.Helper import Helper

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(root_dir)

from biomedisa_extension.SegmentEditorBiomedisa.Logic.BiomedisaLogic import BiomedisaLogic
from biomedisa_extension.SegmentEditorBiomedisa.Logic.BiomedisaParameter import BiomedisaParameter
from biomedisa_extension.SegmentEditorCommon.AbstractBiomedisaSegmentEditorEffect import AbstractBiomedisaSegmentEditorEffect
from biomedisa_extension.SegmentEditorCommon.ListSelectionDialog import ListSelectionDialog

class SegmentEditorEffect(AbstractBiomedisaSegmentEditorEffect):
  """This effect uses the Biomedisa algorithm to segment large 3D volumetric images"""

  def __init__(self, scriptedEffect):
    scriptedEffect.perSegment = False
    scriptedEffect.requireSegments = True
    super().__init__(scriptedEffect, 'Biomedisa Smart Interpolation')

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
      Smart interpolation of sparsely pre-segmented slices<br> creates complete 3D segmentation taking into account the underlying image data. Instructions:</p>
      <ul>
        <li>Create complete pre-segmentation on at least one axial slice (red).</li>
        <li>If you label in different orientations from axial, always leave at least one empty slice between pre-segmented slices.</li>
        <li>All visible segments will be interpolated, not just the selected segment.</li>
        <li><b>All axes:</b> enables pre-segmentation in all orientations.</li>
        <li><b>Denoise:</b> smooths/denoises image data before processing.</li>
        <li><b>Number:</b> number of random walks starting at each pre-segmented pixel.</li>
        <li><b>Steps:</b> steps of each random walk.</li>
        <li><b>Smoothing:</b> number of smoothing iterations for segmentation result.</li>
        <li><b>Remove islands:</b> removes islands/outliers, e.g. 0.5 means that objects smaller than 50 percent of the size of the largest object will be removed.</li>
        <li><b>Fill holes:</b> fills holes, e.g. 0.5 means that all holes smaller than 50 percent of the entire label will be filled.</li>
        <li><b>Ignore:</b> ignores specific label(s), e.g., 2,5,6.</li>
        <li><b>Only:</b> segments only specific label(s), e.g., 1,3,5.</li>
        <li><b>Platform:</b> specifies which platform is used for calculation.</li>
      </ul>
      <p>
        Masking settings are ignored. It is a quasi-interpolation because the pre-segmented slices will be adjusted, and segmentation goes beyond the last pre-segmented slices. The effect uses a <a href="https://doi.org/10.1117/12.2216202">diffusion method</a>. For more information, visit the <a href="https://biomedisa.info/">project page</a>.
      </p>
      <p><u>Contributors:</u> <i>Matthias Fabian, Philipp Lösel</i></p>
      <p>
    </html>"""

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

    # number of random walks
    self.nbrw = qt.QSpinBox()
    self.nbrw.toolTip = 'Number of random walks starting at each pre-segmented pixel'
    self.nbrw.minimum = 1
    self.nbrw.maximum = 1000
    self.nbrw.value = 10
    collapsibleLayout.addRow("Number:", self.nbrw)

    # steps of each random walk
    self.sorw = qt.QSpinBox()
    self.sorw.toolTip = 'Steps of a random walk'
    self.sorw.minimum = 1
    self.sorw.maximum = 1000000
    self.sorw.value = 4000
    collapsibleLayout.addRow("Steps:", self.sorw)

    # smoothing
    self.smooth_active = qt.QCheckBox()
    self.smooth_active.toolTip = 'If deactivated no smoothing will be performed.'
    self.smooth_active.stateChanged.connect(self.onSmoothActiveChanged)

    self.smooth = qt.QSpinBox()
    self.smooth.toolTip = 'Number of smoothing steps.'
    self.smooth.minimum = 0
    self.smooth.maximum = 10000
    self.smooth.value = 100
    self.smooth.setEnabled(False)

    self.smooth_layout = qt.QHBoxLayout()
    self.smooth_layout.addWidget(self.smooth_active)
    self.smooth_layout.addWidget(self.smooth)
    self.smooth_layout.setStretch(0, 0)
    self.smooth_layout.setStretch(1, 1)
    collapsibleLayout.addRow("Smoothing:", self.smooth_layout)

    # remove outliers
    self.clean_active = qt.QCheckBox()
    self.clean_active.toolTip = 'If deactivated no cleaning will be performed.'
    self.clean_active.stateChanged.connect(self.onCleanActiveChanged)

    self.clean = qt.QDoubleSpinBox()
    self.clean.toolTip = 'Remove outliers.'
    self.clean.setRange(0.0, 1.0)   # Set the range
    self.clean.setDecimals(1)       # Set the number of decimals
    self.clean.setSingleStep(0.1)
    self.clean.value = 0.1
    self.clean.setEnabled(False)

    self.clean_layout = qt.QHBoxLayout()
    self.clean_layout.addWidget(self.clean_active)
    self.clean_layout.addWidget(self.clean)
    self.clean_layout.setStretch(0, 0)
    self.clean_layout.setStretch(1, 1)
    collapsibleLayout.addRow("Remove islands:", self.clean_layout)

    # fill holes
    self.fill_active = qt.QCheckBox()
    self.fill_active.toolTip = 'If deactivated no filling will be performed.'
    self.fill_active.stateChanged.connect(self.onFillActiveChanged)

    self.fill = qt.QDoubleSpinBox()
    self.fill.toolTip = 'Fill holes.'
    self.fill.setRange(0.0, 1.0)   # Set the range
    self.fill.setDecimals(1)       # Set the number of decimals
    self.fill.setSingleStep(0.1)
    self.fill.value = 0.9
    self.fill.setEnabled(False)

    self.fill_layout = qt.QHBoxLayout()
    self.fill_layout.addWidget(self.fill_active)
    self.fill_layout.addWidget(self.fill)
    self.fill_layout.setStretch(0, 0)
    self.fill_layout.setStretch(1, 1)
    collapsibleLayout.addRow("Fill holes:", self.fill_layout)

    # ignore labels
    self.ignore = qt.QLineEdit()
    self.ignore.text = ''#'none'
    self.ignore.toolTip = 'Ignore specific label(s), e.g. 2,5,6'
    self.ignore.setPlaceholderText('Enter label(s) to be ignored, e.g. "2,5,6" or "none"...')
    collapsibleLayout.addRow("Ignore:", self.ignore)

    # compute only specific labels
    self.only = qt.QLineEdit()
    self.only.text = ''#'all'
    self.only.toolTip = 'Segment only specific label(s), e.g. 1,3,5'
    self.only.setPlaceholderText('Enter to run only specific label(s), e.g. "1,3,5" or "all"...')
    collapsibleLayout.addRow("Only:", self.only)

    # compute platform
    self.platform = qt.QLineEdit()
    self.platform.text = ''#self.getPlatform()
    self.platform.toolTip = 'One of "cuda", "opencl_NVIDIA_GPU", "opencl_Intel_CPU"'
    self.platform.setPlaceholderText('Enter one of "cuda", "opencl_NVIDIA_GPU", "opencl_Intel_CPU", "None", ...')
    collapsibleLayout.addRow("Platform:", self.platform)

    self.setParameterToGui(BiomedisaParameter())

    collapsibleLayout.addRow("Parameter:", self.createParameterGui())
    AbstractBiomedisaSegmentEditorEffect.setupOptionsFrame(self)

  def getPlatform(self) -> str:
    class Biomedisa:
      def __init__(self, platform=None, success=True, available_devices=0):
        self.platform = platform
        self.success = success
        self.available_devices = available_devices
    try:
      from biomedisa.features.biomedisa_helper import _get_platform
      result = Biomedisa(platform=None, success=True, available_devices=0)
      _get_platform(result)
      if(result.success):
        return result.platform
    except:
      print("No module named 'biomedisa'")

    return ''

  def onLoadParameter(self):
    parameterList = self.getSavedParameter()
    self.dialog = ListSelectionDialog(parameterList)
    def handleDialogClosed(selected_item):
      if selected_item:
        parameter = self.loadParameter(BiomedisaParameter, selected_item)
        self.setParameterToGui(parameter)
    self.dialog.dialogClosed.connect(handleDialogClosed)
    self.dialog.show()

  def onRestoreParameter(self):
    parameter = BiomedisaParameter()
    self.setParameterToGui(parameter)

  def getParameterFromGui(self) -> BiomedisaParameter:
    parameter = BiomedisaParameter()
    parameter.allaxis = self.allaxis.isChecked()
    parameter.denoise = self.denoise.isChecked()
    parameter.nbrw = self.nbrw.value
    parameter.sorw = self.sorw.value
    parameter.smooth_active = self.smooth_active.isChecked()
    parameter.smooth = self.smooth.value
    parameter.clean_active = self.clean_active.isChecked()
    parameter.clean = self.clean.value
    parameter.fill_active = self.fill_active.isChecked()
    parameter.fill = self.fill.value
    parameter.ignore = self.ignore.text if self.ignore.text else 'none'
    parameter.only = self.only.text if self.only.text else 'all'
    parameter.platform = self.platform.text if self.platform.text and self.platform.text != 'None' else None
    return parameter

  def setParameterToGui(self, parameter: BiomedisaParameter):
    self.allaxis.setChecked(parameter.allaxis)
    self.denoise.setChecked(parameter.denoise)
    self.nbrw.value = parameter.nbrw
    self.sorw.value = parameter.sorw
    self.smooth_active.setChecked(parameter.smooth_active)
    self.smooth.value = parameter.smooth
    self.clean_active.setChecked(parameter.clean_active)
    self.clean.value = parameter.clean
    self.fill_active.setChecked(parameter.fill_active)
    self.fill.value = parameter.fill
    self.ignore.text = parameter.ignore
    self.only.text = parameter.only
    self.platform.text = parameter.platform if parameter.platform is not None else 'None'

  def onSmoothActiveChanged(self, state):
    self.smooth.setEnabled(state == qt.Qt.Checked)

  def onCleanActiveChanged(self, state):
    self.clean.setEnabled(state == qt.Qt.Checked)

  def onFillActiveChanged(self, state):
    self.fill.setEnabled(state == qt.Qt.Checked)

  def runAlgorithm(self):
    self.createPreviewNode()

    segmentation = self.previewSegmentationNode.GetSegmentation()
    segmentID = self.scriptedEffect.parameterSetNode().GetSelectedSegmentID()
    segment = segmentation.GetSegment(segmentID)

    # Get modifier labelmap
    binaryLabelmap = segment.GetRepresentation(slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName())
    # Get source volume image data
    volumeNode = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode()
    sourceImageData = volumeNode.GetImageData()
    # Get direction matrix
    direction_matrix = np.zeros((3,3))
    volumeNode.GetIJKToRASDirections(direction_matrix)

    parameter = self.getParameterFromGui()
    # Run the algorithm
    result = BiomedisaLogic.runBiomedisa(input=sourceImageData,
        labels=binaryLabelmap, direction_matrix=direction_matrix, parameter=parameter)

    # Replace voxel data
    binaryLabelmap.SetDimensions(sourceImageData.GetDimensions())
    new_np = result.astype(np.uint16).ravel()
    vtk_arr = numpy_to_vtk(new_np, deep=True, array_type=vtk.VTK_UNSIGNED_SHORT)
    vtk_arr.SetName("ImageScalars")
    binaryLabelmap.GetPointData().SetScalars(vtk_arr)

    # Notify Slicer
    self.previewSegmentationNode.Modified()

    '''for label, binaryLabelmap in enumerate(resultLabelMaps):
      # Get segment ID from label index. This is 0 based even though first the voxel value is 1.
      segmentID = segmentation.GetNthSegmentID(label)
      slicer.vtkSlicerSegmentationsModuleLogic.SetBinaryLabelmapToSegment(
        binaryLabelmap, 
        self.previewSegmentationNode, 
        segmentID, 
        slicer.vtkSlicerSegmentationsModuleLogic.MODE_REPLACE, 
        binaryLabelmap.GetExtent())'''

