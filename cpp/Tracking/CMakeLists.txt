cmake_minimum_required(VERSION 3.24)
project(Tracking LANGUAGES CXX)

set(TDNI_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../VERSION")
file(STRINGS "${TDNI_VERSION_FILE}" TDNI_VERSION LIMIT_COUNT 1)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${TDNI_VERSION_FILE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# The tables must match the Python tracker's text exactly: no FMA contraction, no fast math.
if(MSVC)
	add_compile_options(/fp:precise /W3)
endif()

# The tracker itself (no TouchDesigner dependency): pipelines, matching, table formatting.
add_library(tracking_core STATIC src/KeypointTracker.cpp src/PoseTracker.cpp src/Tracker.cpp)
target_include_directories(tracking_core PUBLIC src)
target_compile_definitions(tracking_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

# C API for tests/tracking_replay.py (ctypes): recorded Info DATs in, tables out.
add_library(tracker_replay SHARED tools/tracker_replay.cpp)
target_link_libraries(tracker_replay PRIVATE tracking_core)

# The Tracker custom DAT (TrackerDAT.dll), copied into <repo>/plugins/.
set(TD_INSTALL_DIR "C:/Program Files/Derivative/TouchDesigner" CACHE PATH "TouchDesigner install folder")
set(TD_SDK_DIR "${TD_INSTALL_DIR}/Samples/CPlusPlus/DAT" CACHE PATH "Folder containing DAT_CPlusPlusBase.h")
if(NOT EXISTS "${TD_SDK_DIR}/DAT_CPlusPlusBase.h")
	message(FATAL_ERROR "DAT_CPlusPlusBase.h not found in TD_SDK_DIR=${TD_SDK_DIR}")
endif()
add_library(TrackerDAT SHARED src/TrackerDAT.cpp)
target_include_directories(TrackerDAT PRIVATE "${TD_SDK_DIR}")
target_link_libraries(TrackerDAT PRIVATE tracking_core)
target_compile_definitions(TrackerDAT PRIVATE TDNI_VERSION="${TDNI_VERSION}")
set(PLUGIN_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../plugins")
add_custom_command(TARGET TrackerDAT POST_BUILD
	COMMAND ${CMAKE_COMMAND} -DSRC=$<TARGET_FILE:TrackerDAT> -DDST=${PLUGIN_OUT_DIR}/$<TARGET_FILE_NAME:TrackerDAT>
		-P "${CMAKE_CURRENT_SOURCE_DIR}/../MediaPipeTOP/cmake/copy_plugin.cmake"
	VERBATIM)
