# CUDA 

The gp-gpu made easy.

## CUDA toolkit
    nvcc compiler
     
    debuggers

    NVIDIA Visual Profilers

## Installation process 

download at :
https://www.nvidia.com/download/driverResults.aspx/169402/en-us

i) `dpkg -i nvidia-driver-local-repo-ubuntu1804-460.32.03_1.0-1_amd64.deb`
ii) `apt-get update`
iii) `apt-get install cuda-drivers`
iv) `reboot`

However there seem to have been conflicts between old drivers

sudo aptitude install cuda-drivers removed old and solved problems

There seems also that you dhould be able to install the drivers directly in the install of tool kit.


## Memory set in RAM
is called host data.

## Memory on GPU
is called device.


## The kernel
with data on the device we can deploy a program or a kernel


## General workflow


generate data -> perform computation -> back to host


    __global__ < tells cuda what is the kernel

global is the function running on the GPU.
So we've got a one unit working on the host and one on the device.
The body of the for loop. Each thread will run the code on each element

## Execution configuration

add\_kernel<<<BS, n_blocks>>>
The number of threads to launch
How to configure these threads into blocks

### Section 1


So general debugging after first lecture. 


    int index = blockIdx.x * blockDim.x + threadIdx.x;

    see what operation we use! this is probably similar to the indexing of an image or similar

### Section 2

# Software stack
    1. Low level API (driver api)
    Shipped along with the driver

    2. High level API (h-file) runtime api
    Included with C



# Execution model

    1. Kernel

    2. Grid[ Block\_0[ Thread\_0 .. ] .. ]


# Hardware Architecture

 ## Streaming Multiprocessors

 ## SIMT unlike SIMD this is scalable

 Warps = 32 threads, can bee seen as one compute unit

 No overhead for switching warps

## Running a kernel

    1. Blocks are assigned to available SMs
    2. Blocks are split into Warps
    3. Multiple warps/blocks run on each SM

## Kernel Execution Configurations


