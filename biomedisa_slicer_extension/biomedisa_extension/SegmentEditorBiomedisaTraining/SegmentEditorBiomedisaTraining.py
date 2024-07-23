import os
import slicer
from slicer.ScriptedLoadableModule import *

# Source: https://github.com/lassoan/SlicerSegmentEditorExtraEffects
class SegmentEditorBiomedisaTraining(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        import string
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SegmentEditorBiomedisaTraining"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = ["Segmentations"]
        self.parent.contributors = ["Matthias Fabian"]
        self.parent.hidden = True
        self.parent.helpText = "This hidden module registers the segment editor effect"
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        slicer.app.connect("startupCompleted()", self.registerEditorEffect)

    def registerEditorEffect(self):
        import qSlicerSegmentationsEditorEffectsPythonQt as qSlicerSegmentationsEditorEffects
        instance = qSlicerSegmentationsEditorEffects.qSlicerSegmentEditorScriptedEffect(None)
        effectFilename = os.path.join(os.path.dirname(__file__), self.__class__.__name__+'Lib/SegmentEditorEffect.py')
        instance.setPythonSource(effectFilename.replace('\\','/'))
        print(f"instance: {instance}")
        print(f"self: {instance.self()}")
        instance.self().register()
        