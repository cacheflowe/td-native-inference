cmake_minimum_required(VERSION 3.24)
project(Tracking LANGUAGES CXX)

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
target_compile_definitions(tracking_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)

# C API for tests/tracking_replay.py (ctypes): recorded Info DATs in, tables out.
add_library(tracker_replay SHARED tools/tracker_replay.cpp)
target_link_libraries(tracker_replay PRIVATE tracking_core)
