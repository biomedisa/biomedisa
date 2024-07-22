import qt, slicer
from SegmentEditorEffects import *

class AbstractBiomedisaSegmentEditorEffect(AbstractScriptedSegmentEditorEffect):
    """This effect uses the Biomedisa algorithm to segment large 3D volumetric images"""

    def __init__(self, scriptedEffect):
        self.previewSegmentationNode = None
        AbstractScriptedSegmentEditorEffect.__init__(self, scriptedEffect)

    def runAlgorithm(self):
        pass

    def setupOptionsFrame(self):
        self.runButton = qt.QPushButton("Initialize")
        self.runButton.objectName = self.__class__.__name__ + 'Run'
        self.runButton.setToolTip("Run the biomedisa algorithm and generate segment data")
        self.runButton.connect('clicked()', self.onRun)
        self.scriptedEffect.addOptionsWidget(self.runButton)

        self.previewShow3DButton = qt.QPushButton("Show 3D")
        self.previewShow3DButton.objectName = self.__class__.__name__ + 'Show3D'
        self.previewShow3DButton.toolTip = "Toggle 3D visibility of the segmentation"
        self.previewShow3DButton.setEnabled(False)
        self.previewShow3DButton.setCheckable(True)
        self.previewShow3DButton.clicked.connect(self.onShow3DButtonClicked)
        self.scriptedEffect.addOptionsWidget(self.previewShow3DButton)

        self.fileLayout = qt.QHBoxLayout()
        self.cancelButton = qt.QPushButton("Cancel")
        self.cancelButton.objectName = self.__class__.__name__ + 'Cancel'
        self.cancelButton.setToolTip("Clear preview and cancel")
        self.cancelButton.setEnabled(False)
        self.cancelButton.connect('clicked()', self.onCancel)
        self.fileLayout.addWidget(self.cancelButton)

        self.selectModelButton = qt.QPushButton("Apply")
        self.selectModelButton.objectName = self.__class__.__name__ + 'Apply'
        self.selectModelButton.setToolTip("Run the biomedisa algorithm and generate segment data")
        self.selectModelButton.setEnabled(False)
        self.selectModelButton.connect('clicked()', self.onApply)
        self.fileLayout.addWidget(self.selectModelButton)
        self.scriptedEffect.addOptionsWidget(self.fileLayout)

    def onCancel(self):
        # delete preview segmentation node
        self.runButton.setEnabled(True)
        self.removePreviewNode()

    def onApply(self):
        # move result form preview nod to main node and delete preview segmentation node
        self.originalSegmentationNode.GetSegmentation().DeepCopy(self.previewSegmentationNode.GetSegmentation())
        self.runButton.setEnabled(True)
        self.removePreviewNode()
    
    def onShow3DButtonClicked(self):
        showing = self.getPreviewShow3D()
        self.setPreviewShow3D(not showing)
    
    def getPreviewNode(self):
        if self.previewSegmentationNode is not None:
            return self.previewSegmentationNode
        return None

    def getPreviewShow3D(self) -> bool:
        previewNode = self.getPreviewNode()
        if not previewNode:
            return False
        containsClosedSurfaceRepresentation = previewNode.GetSegmentation().ContainsRepresentation(
            slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName())
        return containsClosedSurfaceRepresentation

    def setPreviewShow3D(self, show):
        previewNode = self.getPreviewNode()
        if previewNode:
            if show:
                previewNode.CreateClosedSurfaceRepresentation()
                # Refresh the 3D view
                threeDWidget = slicer.app.layoutManager().threeDWidget(0)
                threeDWidget.threeDView().resetFocalPoint()
            else:
                previewNode.RemoveClosedSurfaceRepresentation()

        # Make sure the GUI is up-to-date
        wasBlocked = self.previewShow3DButton.blockSignals(True)
        self.previewShow3DButton.checked = show
        self.previewShow3DButton.blockSignals(wasBlocked)

    def updateButtonStates(self):
        if self.previewSegmentationNode is None:
            self.selectModelButton.setEnabled(False)
            self.cancelButton.setEnabled(False)
            self.previewShow3DButton.setEnabled(False)
        else:
            self.selectModelButton.setEnabled(True)
            self.cancelButton.setEnabled(True)
            self.previewShow3DButton.setEnabled(True)
   
    def createPreviewNode(self):
        self.originalSegmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
        self.previewSegmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.previewSegmentationNode.SetName("Segmentation preview")
        self.previewSegmentationNode.GetSegmentation().DeepCopy(self.originalSegmentationNode.GetSegmentation())

        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationDisplayNode")
        displayNode.SetVisibility3D(True)
        displayNode.SetVisibility2DFill(True)
        displayNode.SetVisibility2DOutline(True)
        self.previewSegmentationNode.SetAndObserveDisplayNodeID(displayNode.GetID())

    def removePreviewNode(self):
        if self.previewSegmentationNode:
            slicer.mrmlScene.RemoveNode(self.previewSegmentationNode)
            self.previewSegmentationNode = None
        self.originalSegmentationNode = None
        self.updateButtonStates()
        
    def onRun(self):
        # This can be a long operation - indicate it to the user
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            slicer.util.showStatusMessage('Running Biomedisa...', 2000)
            self.runButton.setEnabled(False)
            self.scriptedEffect.saveStateForUndo()
            self.runAlgorithm()
            self.updateButtonStates()
            slicer.util.showStatusMessage('Biomedisa finished', 2000)
        except:
            self.runButton.setEnabled(True)
        finally:
            qt.QApplication.restoreOverrideCursor()
