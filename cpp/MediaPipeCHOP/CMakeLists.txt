cmake_minimum_required(VERSION 3.24)
project(MediaPipeCHOP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The release version (VERSION at the repo root) is compiled in: the info popup shows it and
# scripts/release.py checks that packaged plugins carry it.
set(TDNI_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../VERSION")
file(STRINGS "${TDNI_VERSION_FILE}" TDNI_VERSION LIMIT_COUNT 1)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${TDNI_VERSION_FILE}")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# TouchDesigner CHOP plugin SDK headers, read straight from the installed TD.
set(TD_INSTALL_DIR "C:/Program Files/Derivative/TouchDesigner" CACHE PATH "TouchDesigner install folder")
set(TD_SDK_DIR "${TD_INSTALL_DIR}/Samples/CPlusPlus/CHOP" CACHE PATH "Folder containing CHOP_CPlusPlusBase.h")
if(NOT EXISTS "${TD_SDK_DIR}/CHOP_CPlusPlusBase.h")
	message(FATAL_ERROR "CHOP_CPlusPlusBase.h not found in TD_SDK_DIR=${TD_SDK_DIR}")
endif()

# MediaPipe Tasks C API headers, shared with MediaPipeTOP (vendored from tag v1.2.2, matching
# libmediapipe.dll). Nothing is linked: MpAudio.cpp binds the DLL at runtime.
set(MP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../MediaPipeTOP/third_party/mediapipe")

add_library(MediaPipeCHOP SHARED src/MediaPipeCHOP.cpp src/MpAudio.cpp)
target_include_directories(MediaPipeCHOP PRIVATE src "${TD_SDK_DIR}" "${MP_INCLUDE_DIR}")
target_compile_definitions(MediaPipeCHOP PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN TDNI_VERSION="${TDNI_VERSION}")

# Console test: classifies a synthetic sine and silence (no TD).
add_executable(audio_test tools/audio_test.cpp src/MpAudio.cpp)
target_include_directories(audio_test PRIVATE src "${MP_INCLUDE_DIR}")
target_compile_definitions(audio_test PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

# Console test: streams a WAV through the speech-to-text transcriber (no TD).
add_executable(asr_test tools/asr_test.cpp src/MpAudio.cpp)
target_include_directories(asr_test PRIVATE src "${MP_INCLUDE_DIR}")
target_compile_definitions(asr_test PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

# Copy the plugin into <repo>/plugins/, next to libmediapipe.dll (shared with MediaPipeTOP).
set(PLUGIN_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../plugins")
add_custom_command(TARGET MediaPipeCHOP POST_BUILD
	COMMAND ${CMAKE_COMMAND} -DSRC=$<TARGET_FILE:MediaPipeCHOP> -DDST=${PLUGIN_OUT_DIR}/$<TARGET_FILE_NAME:MediaPipeCHOP>
		-P "${CMAKE_CURRENT_SOURCE_DIR}/../MediaPipeTOP/cmake/copy_plugin.cmake"
	VERBATIM)
