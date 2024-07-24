from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QListWidget, QHBoxLayout

class ListSelectionDialog(QDialog):
    def __init__(self, items, parent=None):
        super().__init__(parent)
        
        # Set up the dialog layout
        self.setWindowTitle("Select a parameter set")
        layout = QVBoxLayout()

        # Create and populate the list widget
        self.listWidget = QListWidget()
        self.listWidget.addItems(items)
        layout.addWidget(self.listWidget)
        
        # Create buttons
        buttonLayout = QHBoxLayout()
        self.okButton = QPushButton("OK")
        self.cancelButton = QPushButton("Cancel")
        buttonLayout.addWidget(self.okButton)
        buttonLayout.addWidget(self.cancelButton)
        layout.addLayout(buttonLayout)
        
        # Set the layout for the dialog
        self.setLayout(layout)
        
        # Connect buttons to slot methods
        self.okButton.clicked.connect(self.accept)
        self.cancelButton.clicked.connect(self.reject)

    def getSelectedItem(self):
        return self.listWidget.currentItem().text() if self.listWidget.currentItem() else None
    