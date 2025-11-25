import  slicer, qt, vtk
from slicer import qMRMLNodeComboBox
import numpy as np

class ROISelectionWidget(qt.QWidget):

    def __init__(self, scriptedEffect, baseROIName: str, parent=None):
        super(ROISelectionWidget, self).__init__(parent)
        self.scriptedEffect = scriptedEffect
        self.enforceRoiLimitsSemaphor = False
        self.baseROIName = baseROIName
        self.roiObserverTag = None

        # Create the combo box for selecting an ROI
        self.inputROISelector = qMRMLNodeComboBox()
        self.inputROISelector.nodeTypes = ["vtkMRMLMarkupsROINode"]
        self.inputROISelector.selectNodeUponCreation = True
        self.inputROISelector.addEnabled = True
        self.inputROISelector.removeEnabled = True
        self.inputROISelector.renameEnabled = True
        self.inputROISelector.noneEnabled = True
        self.inputROISelector.showHidden = True
        self.inputROISelector.showChildNodeTypes = False
        self.inputROISelector.setMRMLScene(slicer.mrmlScene)
        self.inputROISelector.setToolTip("Select or create an ROI node.")

        self.visibleIcon = qt.QIcon(":/Icons/Small/SlicerVisible.png")
        self.invisibleIcon = qt.QIcon(":/Icons/Small/SlicerInvisible.png")
        # Create buttons for visibility toggle and fitting to volume
        self.visibilityButton = qt.QToolButton()
        self.visibilityButton.setToolTip("Toggle the visibility of the selected ROI")
        self.visibilityButton.setCheckable(True)
        self.visibilityButton.setIcon(self.visibleIcon)
        self.visibilityButton.setText("Toggle visibility")
        self.visibilityButton.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
        self.visibilityButton.setEnabled(False)

        self.fitToVolumeButton = qt.QToolButton()
        self.fitToVolumeButton.setToolTip("Fit ROI to the volume")
        self.fitToVolumeButton.setIcon(qt.QIcon(":/Icons/ViewCenter.png"))
        self.fitToVolumeButton.setText("Fit to Volume")
        self.fitToVolumeButton.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
        self.fitToVolumeButton.setEnabled(False)

        # Layout
        layout = qt.QVBoxLayout()

        row0 = qt.QHBoxLayout()
        row0.addWidget(self.inputROISelector)

        row1 = qt.QHBoxLayout()
        row1.addWidget(self.visibilityButton)
        row1.addWidget(self.fitToVolumeButton)

        layout.addLayout(row0)
        layout.addLayout(row1)
        self.setLayout(layout)

        # Connect signals to slots
        self.inputROISelector.currentNodeChanged.connect(self.onROISelected)
        self.inputROISelector.nodeAddedByUser.connect(self.onROICreated)
        self.visibilityButton.clicked.connect(self.toggleROIVIsibility)
        self.fitToVolumeButton.clicked.connect(self.fitROIToVolume)

    def onROISelected(self):
        """Handle when a new ROI is selected."""
        if self.roiObserverTag is not None:
            self.removeROIObserver()

        roiNode = self.getSelectedROINode()
        if roiNode:
            self.fitToVolumeButton.setEnabled(True)
            self.visibilityButton.setEnabled(True)
            self.visibilityButton.setChecked(roiNode.GetDisplayVisibility())
            self.updateVisibilityIcon(roiNode.GetDisplayVisibility())
            self.roiObserverTag = roiNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self.enforceRoiLimits)
        else:
            self.fitToVolumeButton.setEnabled(False)
            self.visibilityButton.setEnabled(False)

    def removeROIObserver(self):
        roiNode = self.getSelectedROINode()
        if roiNode and self.roiObserverTag is not None:
            roiNode.RemoveObserver(self.roiObserverTag)
            self.roiObserverTag = None

    def fitROIToVolume(self):
        """Fit the selected ROI to the entire volume."""
        roiNode = self.getSelectedROINode()
        volumeNode = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode()

        if roiNode and volumeNode:
            bounds = [0] * 6
            volumeNode.GetRASBounds(bounds)
            roiNode.SetXYZ([(bounds[1] + bounds[0]) / 2, (bounds[3] + bounds[2]) / 2, (bounds[5] + bounds[4]) / 2])
            roiNode.SetRadiusXYZ([(bounds[1] - bounds[0]) / 2, (bounds[3] - bounds[2]) / 2, (bounds[5] - bounds[4]) / 2])

    def toggleROIVIsibility(self):
        """Toggle the visibility of the selected ROI."""
        roiNode = self.getSelectedROINode()
        if roiNode:
            currentVisibility = roiNode.GetDisplayVisibility()
            newVisibility = not currentVisibility
            roiNode.SetDisplayVisibility(newVisibility)
            self.updateVisibilityIcon(newVisibility)
            self.visibilityButton.setChecked(newVisibility)

    def onROICreated(self):
        if self.roiObserverTag is not None:
            self.removeROIObserver()
        self.fitROIToVolume()
        self.enforceRoiLimits()
        roiNode = self.getSelectedROINode()
        if roiNode:
            self.setUniqueROIName(roiNode, self.baseROIName)
            self.roiObserverTag = roiNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self.enforceRoiLimits)

    def setUniqueROIName(self, roiNode, baseName="ROI"):
        # Iterate over the nodes directly to collect names
        nodeNames = set()
        allROINodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsROINode")

        for i in range(allROINodes.GetNumberOfItems()):
            node = allROINodes.GetItemAsObject(i)
            if node != roiNode:  # Exclude the current node itself
                nodeNames.add(node.GetName())

        # Explicitly delete the vtkCollection to dispose of it
        allROINodes.UnRegister(None)

        # Automatically generate a unique name
        uniqueName = baseName
        suffix = 1
        while uniqueName in nodeNames:
            uniqueName = f"{baseName}_{suffix}"
            suffix += 1

        # Set the unique name to the ROI node
        roiNode.SetName(uniqueName)

    def updateVisibilityIcon(self, isVisible):
        """Update the visibility icon based on the current visibility state."""
        if isVisible:
            self.visibilityButton.setIcon(self.visibleIcon)
        else:
            self.visibilityButton.setIcon(self.invisibleIcon)

    def getSelectedROINode(self):
        """Get the currently selected ROI node."""
        return self.inputROISelector.currentNode()

    def getXYZMinMax(self):
        roiNode = self.getSelectedROINode()
        volumeNode = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode()
        if not roiNode:
            sourceImageData = volumeNode.GetImageData()
            dim = sourceImageData.GetDimensions()
            return [0, dim[0]-1, 0, dim[1]-1, 0, dim[2]-1]
        RASToIJK = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(RASToIJK)
        radius = [0.0, 0.0, 0.0]
        roiNode.GetRadiusXYZ(radius)
        radius.append(0.0)
        radius_ijk = [0.0, 0.0, 0.0, 0.0]
        RASToIJK.MultiplyPoint(radius, radius_ijk)
        radius_ijk = [abs(radius_ijk[0]), abs(radius_ijk[1]), abs(radius_ijk[2])]
        roiCenter = [0.0, 0.0, 0.0]
        roiNode.GetXYZ(roiCenter)
        roiCenter.append(1.0)
        roiCenter_ijk = [0.0, 0.0, 0.0, 0.0]
        RASToIJK.MultiplyPoint(roiCenter, roiCenter_ijk)
        return self.roiToXYZMinMax(radius_ijk, roiCenter_ijk[:-1])

    def setXYZMinMax(self, x_min, x_max, y_min, y_max, z_min, z_max, parameterName):
        roiNode = self.getSelectedROINode()
        if not roiNode and not x_min:
            return
        elif not x_min:
            self.fitROIToVolume()
            return
        elif not roiNode and x_min:
            roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", parameterName)
            self.inputROISelector.setCurrentNode(roiNode)
        volumeNode = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode()
        radius_ijk, roiCenter_ijk = self.xYZMinMaxToRoi(x_min, x_max, y_min, y_max, z_min, z_max)

        IJKToRAS = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(IJKToRAS)

        radius_ijk = [radius_ijk[0], radius_ijk[1], radius_ijk[2], 0.0]
        radius = [0.0, 0.0, 0.0, 0.0]
        IJKToRAS.MultiplyPoint(radius_ijk, radius)
        radius = [abs(radius[0]), abs(radius[1]), abs(radius[2])]

        roiCenter_ijk = [roiCenter_ijk[0], roiCenter_ijk[1], roiCenter_ijk[2], 1.0]
        roiCenter = [0.0, 0.0, 0.0, 0.0]
        IJKToRAS.MultiplyPoint(roiCenter_ijk, roiCenter)

        roiNode.SetRadiusXYZ(radius)
        roiNode.SetXYZ(roiCenter[:-1])

    def roiToXYZMinMax(self, radius, center):
        x_min = center[0] - radius[0]
        x_max = center[0] + radius[0]
        y_min = center[1] - radius[1]
        y_max = center[1] + radius[1]
        z_min = center[2] - radius[2]
        z_max = center[2] + radius[2]
        values = [np.floor(x_min), np.ceil(x_max), np.floor(y_min), np.ceil(y_max), np.floor(z_min), np.ceil(z_max)]
        return [int(x) for x in values] # Cast to int

    def xYZMinMaxToRoi(self, x_min, x_max, y_min, y_max, z_min, z_max):
        x_radius = (x_max - x_min) / 2
        y_radius = (y_max - y_min) / 2
        z_radius = (z_max - z_min) / 2
        x_center = (x_min + x_radius)
        y_center = (y_min + y_radius)
        z_center = (z_min + z_radius)
        return ([x_radius, y_radius, z_radius], [x_center, y_center, z_center])

    def enforceRoiLimits(self, caller=None, event=None):
        roiNode = self.getSelectedROINode()
        if not roiNode:
            return

        if self.enforceRoiLimitsSemaphor:
            return
        self.enforceRoiLimitsSemaphor = True
        volumeNode = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode()

        MINIMUM_SIZE_ijk = [63.0, 63.0, 63.0, 0.0]  # Minimum size along X, Y, Z axes

        IJKToRAS = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(IJKToRAS)
        MINIMUM_SIZE = [0.0, 0.0, 0.0, 0.0]
        IJKToRAS.MultiplyPoint(MINIMUM_SIZE_ijk, MINIMUM_SIZE)
        MINIMUM_SIZE = [abs(MINIMUM_SIZE[0]), abs(MINIMUM_SIZE[1]), abs(MINIMUM_SIZE[2])]

        # Get the volume bounds
        volumeNode = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode()
        bounds = [0] * 6
        volumeNode.GetRASBounds(bounds)
        maxRadius = [(bounds[1] - bounds[0]) / 2, (bounds[3] - bounds[2]) / 2, (bounds[5] - bounds[4]) / 2]

        # Get current ROI radius and center
        radius = [0, 0, 0]
        roiNode.GetRadiusXYZ(radius)

        center = [0, 0, 0]
        roiNode.GetXYZ(center)

        # Enforce minimum size and keep within volume bounds
        for i in range(3):
            radius[i] = max(MINIMUM_SIZE[i] / 2.0, min(radius[i], maxRadius[i]))

            # Adjust the center if the ROI goes out of bounds
            minCenter = bounds[2 * i] + radius[i]
            maxCenter = bounds[2 * i + 1] - radius[i]
            center[i] = max(minCenter, min(center[i], maxCenter))

        # Apply the adjusted size and center back to the ROI
        roiNode.SetRadiusXYZ(radius)
        roiNode.SetXYZ(center)
        self.enforceRoiLimitsSemaphor = False

