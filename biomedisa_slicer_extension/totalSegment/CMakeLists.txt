cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(Segment anything)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/totalSegment")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Matthias Fabian")
set(EXTENSION_DESCRIPTION "bla")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/totalSegment.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/totalSegment/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(biomedisa_segmentation)
add_subdirectory(segment_anything)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
