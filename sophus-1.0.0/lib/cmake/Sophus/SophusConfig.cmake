################################################################################
# Sophus source dir
set( Sophus_SOURCE_DIR "")

################################################################################
# Sophus build dir
set( Sophus_DIR "")

################################################################################
# Compute paths
get_filename_component(Sophus_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

set( Sophus_INCLUDE_DIR  "${Sophus_CMAKE_DIR}/../../../include" )
set( Sophus_INCLUDE_DIRS  "${Sophus_CMAKE_DIR}/../../../include" )
