import slicer
from vtk import vtkCommand, vtkInteractorStyleUser
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication
from dimension_manager import DimensionManager

class MouseEventHandler():
    def __init__(self, color):
        self.color = color
        self.slice_widget = slicer.app.layoutManager().sliceWidget(color)
        slice_view = self.slice_widget.sliceView()
        self.observerId = self.interactor = slice_view.interactorStyle().GetInteractor()
        self.interactor_style = vtkInteractorStyleUser()
        self.coords = []
        self.dimension_manager = DimensionManager(self.color)

    def dimension_manager(self):
        return self.dimension_manager
    
    def __del__(self):
        # Destructor to clean up observers
        self.remove_observers()

    def cleanup(self):
        self.interactor.RemoveObserver(self.observerId)
        
    def setup(self):
        self.observerId = self.interactor.AddObserver(vtkCommand.LeftButtonPressEvent, self.on_left_click)

    def on_left_click(self, caller, event):
        coords = self.get_image_coordinates()
        modifiers = QGuiApplication.queryKeyboardModifiers()
        alt_pressed = modifiers & Qt.Modifier.ALT
        self.addCoord(coords, not alt_pressed)

    def get_image_coordinates(self):
        infoWidget = slicer.modules.DataProbeInstance.infoWidget
        text = infoWidget.layerIJKs['B'].text
        return self.ijkStringToCoordArray(text)
    
    def ijkStringToCoordArray(self, text):
        """Converts the string from slicer's Data Probe to an array of coords."""
        numbers_str = text.strip("() ").split(", ")
        numbers_uint8 = np.array(numbers_str, dtype=np.uint8)
        numbers_uint8 = numbers_uint8.reshape(-1, 1)
        return numbers_uint8
    
    def addCoord(self, coords, isForeground):
        print(f"add {coords} isForeground: {isForeground}")
        coords = [int(coord) for coord in coords]
        self.coords.append([coords[0], coords[1], coords[2], isForeground])

    def getCoords(self, index):
        # TODO: Use dimension manager to determine by which column to filter
        filtered_list = [item for item in self.coords if item[2] == index]
        return filtered_list
    
    def getForegroundCoords(self, index):
        return self.dimension_manager.getForegroundCoords(self.coords, index)
    
    def getBackgroundCoords(self, index):
        return self.dimension_manager.getBackgroundCoords(self.coords, index)
    
    def getCurrentIndex(self) -> int:
        sliceController = self.slice_widget.sliceController()
        sliceValue = sliceController.sliceOffsetSlider().value
        return int(sliceValue)

