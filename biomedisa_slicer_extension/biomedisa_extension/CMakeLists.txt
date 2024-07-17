cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(Biomedisa Label)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/biomedisa_extension")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Matthias Fabian")
set(EXTENSION_DESCRIPTION "Biomedisa is a free and easy-to-use open-source application for segmenting large volumetric images such as CT and MRI scans,
developed at The Australian National University CTLab. Biomedisa's smart interpolation of sparsely pre-segmented slices
enables accurate semi-automated segmentation by considering the complete underlying image data. 
For more information visit the <a href="https://biomedisa.info/">project page</a>")
set(EXTENSION_ICONURL "https://biomedisa.info/static/biomedisa_logo.svg")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/biomedisa_extension/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SegmentEditorBiomedisa)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
