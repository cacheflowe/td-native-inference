cmake_minimum_required(VERSION 3.24)
project(OnnxTOP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# Turing / Ampere / Ada SASS, plus Ada PTX so newer GPUs (Blackwell) can JIT.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES "75-real;86-real;89")
endif()

# TouchDesigner plugin SDK headers (CPlusPlus_Common.h, TOP_CPlusPlusBase.h) are read
# straight from the installed TD so the plugin always matches the TD it runs in.
set(TD_INSTALL_DIR "C:/Program Files/Derivative/TouchDesigner" CACHE PATH "TouchDesigner install folder")
set(TD_SDK_DIR "${TD_INSTALL_DIR}/Samples/CPlusPlus/CudaTOP" CACHE PATH "Folder containing TOP_CPlusPlusBase.h")
if(NOT EXISTS "${TD_SDK_DIR}/TOP_CPlusPlusBase.h")
	message(FATAL_ERROR "TOP_CPlusPlusBase.h not found in TD_SDK_DIR=${TD_SDK_DIR}")
endif()

# ONNX Runtime: headers only (vendored, matching TD's bundled ORT 1.23). Nothing is linked --
# OrtLoader binds to TD's own onnxruntime.dll at runtime.
set(ORT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include")

add_library(onnx_core STATIC
	src/OrtLoader.cpp
	src/OnnxEngine.cpp
	src/YoloDecoder.cpp
	src/TrtBuilder.cpp
	src/Kernels.cu
)
find_package(CUDAToolkit REQUIRED)
target_include_directories(onnx_core PUBLIC src "${ORT_INCLUDE_DIR}")
target_compile_definitions(onnx_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
# cudart64_12.dll (shared) -- TD ships it next to its exe, so the plugin uses TD's copy.
target_link_libraries(onnx_core PUBLIC CUDA::cudart)
set_target_properties(onnx_core PROPERTIES
	POSITION_INDEPENDENT_CODE ON
	CUDA_RUNTIME_LIBRARY Shared)

add_library(OnnxTOP SHARED src/OnnxTOP.cpp)
target_include_directories(OnnxTOP PRIVATE "${TD_SDK_DIR}")
target_link_libraries(OnnxTOP PRIVATE onnx_core)
set_target_properties(OnnxTOP PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# Companion exe: out-of-process TensorRT engine builder (launched by the TOP) + bench.
add_executable(onnxtop_tool tools/onnxtop_tool.cpp)
target_link_libraries(onnxtop_tool PRIVATE onnx_core)
set_target_properties(onnxtop_tool PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# Copy the plugin + its builder exe into <repo>/plugins/ (point the CPlusPlus TOP's Plugin
# Path there). The TOP looks for onnxtop_tool.exe next to its own DLL. Warns instead of
# failing while TD has the DLL loaded -- toggle Unload Plugin first.
set(PLUGIN_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../plugins")
foreach(tgt OnnxTOP onnxtop_tool)
	add_custom_command(TARGET ${tgt} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -DSRC=$<TARGET_FILE:${tgt}> -DDST=${PLUGIN_OUT_DIR}/$<TARGET_FILE_NAME:${tgt}>
			-P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_plugin.cmake"
		VERBATIM)
endforeach()
