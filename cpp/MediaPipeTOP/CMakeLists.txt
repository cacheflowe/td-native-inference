cmake_minimum_required(VERSION 3.24)
project(MediaPipeTOP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The release version (VERSION at the repo root) is compiled in: the info popup shows it and
# scripts/release.py checks that packaged plugins carry it.
set(TDNI_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../VERSION")
file(STRINGS "${TDNI_VERSION_FILE}" TDNI_VERSION LIMIT_COUNT 1)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${TDNI_VERSION_FILE}")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES "75-real;86-real;89")
endif()

# TouchDesigner plugin SDK headers, read straight from the installed TD (same as OnnxTOP).
set(TD_INSTALL_DIR "C:/Program Files/Derivative/TouchDesigner" CACHE PATH "TouchDesigner install folder")
set(TD_SDK_DIR "${TD_INSTALL_DIR}/Samples/CPlusPlus/CudaTOP" CACHE PATH "Folder containing TOP_CPlusPlusBase.h")
if(NOT EXISTS "${TD_SDK_DIR}/TOP_CPlusPlusBase.h")
	message(FATAL_ERROR "TOP_CPlusPlusBase.h not found in TD_SDK_DIR=${TD_SDK_DIR}")
endif()

# MediaPipe Tasks C API: headers only (vendored from google-ai-edge/mediapipe tag v1.2.2, matching
# the libmediapipe.dll that fetch_mediapipe.ps1 installs). Nothing is linked -- MpApi.cpp binds to
# the DLL at runtime with LoadLibrary/GetProcAddress.
set(MP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mediapipe")

# The GPU resize kernel is shared with OnnxTOP (same TD texture formats, same flip convention).
set(ONNXTOP_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../OnnxTOP/src")

find_package(CUDAToolkit REQUIRED)
add_library(MediaPipeTOP SHARED
	src/MediaPipeTOP.cpp
	src/MpApi.cpp
	src/MpWorker.cpp
	src/MpKernels.cu
	"${ONNXTOP_SRC}/Kernels.cu"
)
target_include_directories(MediaPipeTOP PRIVATE src "${ONNXTOP_SRC}" "${TD_SDK_DIR}" "${MP_INCLUDE_DIR}")
target_compile_definitions(MediaPipeTOP PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN TDNI_VERSION="${TDNI_VERSION}")
target_link_libraries(MediaPipeTOP PRIVATE CUDA::cudart)
set_target_properties(MediaPipeTOP PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# Console test: runs MpWorker on raw RGB frames (no TD) to check results against the Python API.
add_executable(mp_test tools/mp_test.cpp src/MpApi.cpp src/MpWorker.cpp)
target_include_directories(mp_test PRIVATE src "${MP_INCLUDE_DIR}")
target_compile_definitions(mp_test PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

# Copy the plugin into <repo>/plugins/ next to OnnxTOP.dll; libmediapipe.dll must sit beside it.
set(PLUGIN_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../plugins")
add_custom_command(TARGET MediaPipeTOP POST_BUILD
	COMMAND ${CMAKE_COMMAND} -DSRC=$<TARGET_FILE:MediaPipeTOP> -DDST=${PLUGIN_OUT_DIR}/$<TARGET_FILE_NAME:MediaPipeTOP>
		-P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_plugin.cmake"
	VERBATIM)
