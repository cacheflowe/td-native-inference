cmake_minimum_required(VERSION 3.24)
project(Puppet LANGUAGES CXX)

set(TDNI_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../VERSION")
file(STRINGS "${TDNI_VERSION_FILE}" TDNI_VERSION LIMIT_COUNT 1)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${TDNI_VERSION_FILE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

if(MSVC)
	add_compile_options(/fp:precise /W3)
endif()

# The solver (no TouchDesigner dependency): rig, targets, solve, place, the cook.
add_library(puppet_core STATIC src/Retarget.cpp src/Puppet.cpp)
target_include_directories(puppet_core PUBLIC src)
target_compile_definitions(puppet_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

# C API for tests/test_puppet_cpp.py (ctypes).
add_library(puppet_replay SHARED tools/puppet_replay.cpp)
target_link_libraries(puppet_replay PRIVATE puppet_core)

# The Puppet custom CHOP (PuppetCHOP.dll), copied into <repo>/plugins/.
set(TD_INSTALL_DIR "C:/Program Files/Derivative/TouchDesigner" CACHE PATH "TouchDesigner install folder")
set(TD_SDK_DIR "${TD_INSTALL_DIR}/Samples/CPlusPlus/CHOP" CACHE PATH "Folder containing CHOP_CPlusPlusBase.h")
if(NOT EXISTS "${TD_SDK_DIR}/CHOP_CPlusPlusBase.h")
	message(FATAL_ERROR "CHOP_CPlusPlusBase.h not found in TD_SDK_DIR=${TD_SDK_DIR}")
endif()
add_library(PuppetCHOP SHARED src/PuppetCHOP.cpp)
target_include_directories(PuppetCHOP PRIVATE "${TD_SDK_DIR}")
target_link_libraries(PuppetCHOP PRIVATE puppet_core)
target_compile_definitions(PuppetCHOP PRIVATE TDNI_VERSION="${TDNI_VERSION}")
set(PLUGIN_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../plugins")
add_custom_command(TARGET PuppetCHOP POST_BUILD
	COMMAND ${CMAKE_COMMAND} -DSRC=$<TARGET_FILE:PuppetCHOP> -DDST=${PLUGIN_OUT_DIR}/$<TARGET_FILE_NAME:PuppetCHOP>
		-P "${CMAKE_CURRENT_SOURCE_DIR}/../MediaPipeTOP/cmake/copy_plugin.cmake"
	VERBATIM)
