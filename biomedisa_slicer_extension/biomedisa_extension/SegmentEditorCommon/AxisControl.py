import qt
import ctk

class AxisControl(qt.QWidget):
    def __init__(self, parent=None):
        super(AxisControl, self).__init__(parent)

        # Create controls
        self.minSpinBox = qt.QSpinBox()
        self.minSpinBox.setRange(0, 4096)
        self.minSpinBox.setValue(0)

        self.maxSpinBox = qt.QSpinBox()
        self.maxSpinBox.setRange(0, 4096)
        self.maxSpinBox.setValue(4096)

        self.rangeSlider = ctk.ctkRangeSlider()
        self.rangeSlider.setOrientation(qt.Qt.Horizontal)
        self.rangeSlider.setMinimum(0)
        self.rangeSlider.setMaximum(4096)
        self.rangeSlider.setMinimumValue(0)
        self.rangeSlider.setMaximumValue(4096)

        # Layout
        layout = qt.QHBoxLayout()
        layout.addWidget(self.minSpinBox)
        layout.addWidget(self.rangeSlider)
        layout.addWidget(self.maxSpinBox)
        self.setLayout(layout)

        # Connect signals and slots
        self.minSpinBox.valueChanged.connect(self.onMinSpinBoxChanged)
        self.maxSpinBox.valueChanged.connect(self.onMaxSpinBoxChanged)
        self.rangeSlider.minimumPositionChanged.connect(self.onSliderMinChanged)
        self.rangeSlider.maximumPositionChanged.connect(self.onSliderMaxChanged)
        
    def onMinSpinBoxChanged(self, value):
        if value > self.maxSpinBox.value:
            self.maxSpinBox.setValue(value)
        self.rangeSlider.setMinimumValue(value)
        
    def onMaxSpinBoxChanged(self, value):
        if value < self.minSpinBox.value:
            self.minSpinBox.setValue(value)
        self.rangeSlider.setMaximumValue(value)
        
    def onSliderMinChanged(self, value):
        self.minSpinBox.setValue(value)
        
    def onSliderMaxChanged(self, value):
        self.maxSpinBox.setValue(value)

    def updateMaximum(self, maximum):
        setValueToMax = True if self.rangeSlider.maximumValue == self.rangeSlider.maximum else False
        self.minSpinBox.setRange(0, maximum)
        self.maxSpinBox.setRange(0, maximum)
        self.rangeSlider.setMinimum(0)
        self.rangeSlider.setMaximum(maximum)
        if setValueToMax:
            self.maxSpinBox.setValue(maximum)

    def getMinValue(self):
        return self.minSpinBox.value
    
    def getMaxValue(self):
        return self.maxSpinBox.value
    
    def setMinValue(self, value):
        if value:
            self.minSpinBox.value = value
    
    def setMaxValue(self, value):
        if value:
            self.maxSpinBox.value = value
    
    def setValues(self, min_value, max_value):
        self.minSpinBox.setValue(min_value)
        self.maxSpinBox.setValue(max_value)
        self.rangeSlider.setMinimumValue(min_value)
        self.rangeSlider.setMaximumValue(max_value)
