cmake_minimum_required(VERSION 3.14)
project(cudaKernelLib VERSION 0.0.0 LANGUAGES CUDA CXX)


find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
# find_package(CUDA REQUIRED)
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
    -O3
)

set(CUDA_HOST_COMPLIER /usr/bin/g++)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# cuda utils
add_library(utils INTERFACE)
target_sources(utils INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/utils/utils.cu)
target_include_directories(utils INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/utils/
	)

add_library(kernels INTERFACE)
# TODO:
# https://stackoverflow.com/questions/48274058/how-do-i-set-cuda-architecture-to-compute-50-and-sm-50-from-cmake-3-10-version
set_property(TARGET kernels PROPERTY CUDA_ARCHITECTURES 61)
target_sources(kernels INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/kernels/functions.cu)
target_include_directories(kernels INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/kernels/
	)

target_link_libraries(kernels INTERFACE
	utils
	)


add_executable(
	${PROJECT_NAME}
	main.cpp
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61)

target_include_directories(${PROJECT_NAME} INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}/kernels/
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

set_target_properties(${PROJECT_NAME} 
	PROPERTIES
	CUDA_SEPERABLE_COMPILATION ON
)

target_link_libraries(${PROJECT_NAME}
	PRIVATE
	kernels
)

