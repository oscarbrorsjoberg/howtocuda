cmake_minimum_required(VERSION 3.17)
project(cudaKernelLib VERSION 0.0.0 LANGUAGES CUDA CXX)

# to create CompileCommands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(cuda_debug_flags 
    "")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)

    set(cuda_debug_flags 
        "-g" # debug host
        "-G" # debug device
        )
elseif(CMAKE_BUILD_TYPE EQUAL Profiling)
    set(cuda_debug_flags 
        "-lineinfo" 
        # -G turns of optimization for device code, 
        # this optimizes but keeps line information
        )
endif()

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

# should not use?
# find_package(CUDA REQUIRED)
# https://stackoverflow.com/questions/27018340/cmake-does-not-properly-find-cuda-library

find_package(CUDAToolkit REQUIRED)

## GLFW backend
add_library(util_GLFWState INTERFACE)
target_sources(util_GLFWState INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/glfw_state/glfw_state.cpp
)
target_include_directories(util_GLFWState INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/glfw_state/
)
target_link_libraries(util_GLFWState 
    INTERFACE
    glfw
)

## DearImgui
add_library(DearImgui INTERFACE)
target_sources(DearImgui INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/imgui/imgui_impl_opengl3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/imgui/imgui_impl_glfw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/imgui/imgui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/imgui/imgui_demo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/imgui/imgui_widgets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/imgui/imgui_draw.cpp
)
target_include_directories(DearImgui INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/Thirdparty/imgui/
)

## imguiBackend
add_library(util_imguiBackend INTERFACE)
target_sources(util_imguiBackend INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/imgui/imguiBackend.cpp
)
target_include_directories(util_imguiBackend INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/imgui/
		${OPENGL_INCLUDE_DIR}
)
target_link_libraries(util_imguiBackend 
    INTERFACE
    DearImgui
)


# What does this do?
enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

set(build_flags #APPEND
 "-std=c++17"
 "-Wall"
 "-Wextra"
 "-pedantic"
)

# CUDA COMPLIER TODO: read this:
# https://developer.nvidia.com/blog/building-cuda-applications-cmake/
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html

set(CUDA_NVCC_FLAGS 
    ${CUDA_NVCC_FLAGS}
    ${cuda_debug_flags}
    -O3
)
set(CUDA_HOST_COMPLIER /usr/bin/g++)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# cuda utils
add_library(utils INTERFACE)
set_property(TARGET utils PROPERTY CUDA_ARCHITECTURES 61)
target_sources(utils INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/utils/utils.cu)
target_include_directories(utils INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/utils/
  # for simple print function
	/home/oscar/sw/gpgpu/cuda-samples/Common/
	)
target_link_libraries(
	utils INTERFACE
	CUDA::cudart
	CUDA::cuda_driver
)

# timer
add_library(timer INTERFACE)
set_property(TARGET timer PROPERTY CUDA_ARCHITECTURES 61)
target_sources(timer INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/kernels/KernelTimer.cu)
target_include_directories(timer INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/kernels/
)

# image reading
add_library(image_reading INTERFACE)
set_property(TARGET image_reading PROPERTY CUDA_ARCHITECTURES 61)
target_sources(image_reading INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/utils/ims/ppm.cu)
target_include_directories(image_reading INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/utils/
)

# TODO:
# https://stackoverflow.com/questions/48274058/how-do-i-set-cuda-architecture-to-compute-50-and-sm-50-from-cmake-3-10-version
# general kernel library -- (functions)

add_library(kernels INTERFACE)
set_property(TARGET kernels PROPERTY CUDA_ARCHITECTURES 61)
target_sources(kernels INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/kernels/functions.cu)
target_include_directories(kernels INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/kernels/
	)
target_link_libraries(kernels INTERFACE
	utils
)

# ----- print cuda
add_executable(
	print_cuda
	app/print_info/main.cpp
)

# set_property(TARGET print_cuda PROPERTY CUDA_ARCHITECTURES 61)
target_include_directories(print_cuda INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/utils/
)

set_property(TARGET print_cuda PROPERTY CUDA_ARCHITECTURES 61)

set_target_properties(print_cuda 
	PROPERTIES
	CUDA_SEPERABLE_COMPILATION ON
)

target_link_libraries(print_cuda
	PRIVATE
	utils
)


# ---- spotlight kernel
add_executable(
  spotlight
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/spotlight.cu
)
set_target_properties(spotlight 
	PROPERTIES
	CUDA_SEPERABLE_COMPILATION ON
        CUDA_ARCHITECTURES 61
)
target_include_directories(spotlight INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/utils/
)

target_link_libraries(spotlight
	PRIVATE
	utils
	timer
        image_reading
)

#  ----- add kernel
add_executable(
	add_kernel
	app/add_kernel/main.cpp
)

# set_property(TARGET add_kernel PROPERTY CUDA_ARCHITECTURES 61)
set_target_properties(add_kernel 
	PROPERTIES
	CUDA_SEPERABLE_COMPILATION ON
        CUDA_ARCHITECTURES 61
)

target_include_directories(add_kernel INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/kernels/
	${CMAKE_CURRENT_SOURCE_DIR}/utils/
)

target_link_libraries(add_kernel
	PRIVATE
	kernels
	utils
	timer
)


# --- test imgui GLFW
add_executable(
    ui_window
    ${CMAKE_CURRENT_SOURCE_DIR}/app/ui_window/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/opengl/gl.cpp
)

set_target_properties(
    ui_window
    PROPERTIES
    CUDA_SEPERABLE_COMPILATION ON
    CUDA_ARCHITECTURES 61
)

target_include_directories(
    ui_window 
    PRIVATE ## this can't be interface
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/opengl/
    ${OPENGL_INCLUDE_DIR}
)

target_link_libraries(
    ui_window
    PRIVATE
    util_GLFWState
    util_imguiBackend
    GLEW::glew 
    OpenGL::GL
    glfw
)

# main will be build with build flags as above,
# public might unessesary since the nothing will
# link to main
# target_compile_features(${PROJECT_NAME} 
# 	PUBLIC
# 	${build_flags}
# 	)
# Same as -dc flag
# Compile each .c, .cc, .cpp, .cxx, and .cu input file into an object file that contains 
# relocatable device code.

# set_target_properties(add_kernel 
# 	PROPERTIES
# 	CUDA_SEPERABLE_COMPILATION ON
# )


