# CUDA 

The gp-gpu made easy.


## Programming with CUDA

General overview of programming Model

### CUDA Software stack -- driver and runtime API

1. Driver API
Low level
2. Runtime API
The cuda toolkit includes the runtime api, see header files
	
### Execution model -- kernel, threads and blocks

Kernel launced from host and runs on device.
N CUDA thread on device.
Threads are grouped into blocks, and blocks are arranged into a grid.
Threads can communicate inbetween blocks.
Several SMs on each GPU
Multiple CUDA cores per SM
Shared cache, registers and memory between cores
Global memory for all.
No branch prediction or speculations.
SM{Warp[ 32 threads]}


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


## The kernel / same as shader program
with data on the device we can deploy a program or a kernel


## General workflow

generate data -> perform computation -> back to host


    __global__ < tells cuda what is the kernel

global is the function running on the GPU.
So we've got a one unit working on the host and one on the device.
The body of the for loop. Each thread will run the code on each element


## Execution configuration

add\_kernel<<<BS, n_blocks>>>
BS : The number of threads to launch
n\_blocks : How to configure these threads into blocks

### Section 1


So general debugging after first lecture. 


    int index = blockIdx.x * blockDim.x + threadIdx.x;

# Section 2

## Software stack
    1. Low level API (driver api)
    Shipped along with the driver

    2. High level API (h-file) runtime api
    Included with C

## Execution model

    // the code
    1. Kernel

    // the data structure
    2. Grid [ Block\_0[ Thread\_0 .. ] .. ]

    // the operators
    3. Streaming Multiprocessors (SMs)

    4. Stream Processors (SPs)



Kernel-> sets up grid and block size.
SM -> executes block according to the kernel code.
SP -> executes thread in block according to kernel code.

An SM can handle a warp at a time (32 threads).


## Hardware Architecture

 ## Streaming Multiprocessors

 ## SIMT unlike SIMD this is scalable

 Warps = 32 threads, can bee seen as one compute unit

 No overhead for switching warps

## Running a kernel

    1. Blocks are assigned to available SMs
    2. Blocks are split into Warps
    3. Multiple warps/blocks run on each SM

## Kernel Execution Configurations

my\_kernel<<< GRID\_DIMENSION, BLOCK\_DIMENSION (number of threads in each block)>>>


### Working with 2d cases:
To better handle dimensions in the cuda framework, ie go from acctual dimensions to the cuda dimensions Blocks and Threads,
One should use the 'dim3' command.


## How to choose a block size? 

It all comes down to performance.

Occupancy -- keeping all the SMs busy.

Sync and communication.

### Rules of thumb

Fill upp our warps (size 32)

Either 128 or 256

More blocks than SM to avoid SM idle

Several smaller blocks are often better than few large ones

Always measure the performance!

Occupancy = warps/maximum

Higher is better

Maximize with the occupancy API:

'''
 cudaOccupancyMaxPotentialBlockSize();
'''


### cuda-gdb

nvcc 
-g == for debugging like normal
-G == for device code


to get it running
to break device code
CUDA\_DEBUGGER\_SOFTWARE\_PREEMPTION=1 cuda-gdb


To change to othet block/thread


For a more brute force alternative, we should be able to 
''' set cuda memcheck on '''

### handling errors

    1. cudaError\_t << error type
    2. cudaGetErrorString() - human readable, 
    3. cudaGetLastError()
    4.  
    	status = cudaDeviceSynchronize();
    	message = cudaGetErrorString(status);

 	Get the error 

It can be wise to write some type of wrapper for the error handling.  


### handling kernel errors
    1. cudaDeviceSynchronize() - waits for kernel to finish and returns error
    This might cause problems later, since the state rests in the system break the context


# Section 3 Performance Optimizations

Since the performance is what we're looking for, its important to optimize the code to fit the context.

## Section 3.1  NVIDIA Visual Profiler

Trace device code and CUDA API calls
Guided analysis

So the general nvvp is quite problematic.
There seems that there is a need to change nsys-ui.

There is quite a few profiling tools:
nvvp, nvprof, nsys.

It seems that the new way forward should be nsight.
However the tutorial Im currently watching they suggest the nvvp.
The nvvp seems to be constructed with java8 and is quite a hassel to get up and running on Linux.
There seems to be some kind of problem.

https://developer.nvidia.com/blog/migrating-nvidia-nsight-tools-nvvp-nvprof/


1. Start with nsight systems (this seems to be nsys)
2. Nsight compute -> Cuda kernel optimazation (ncu) (ncu-gui)
3. Nisght graphics -> quite the same, but with other API's

So all of these three binaries should in some way replace the old nvvp.
It was quite a hassel to get up and running and it seems that 
there is no future support for it.

All these tools seems pretty important I will get back to them.

### Nsight systems

https://www.youtube.com/watch?v=kKANP0kL_hk

Profiler output. 
First we want to profile an application.

Create an report with

´´´
    nsys profile
´´´

This report can than be investigated in the nsys-ui

In the drop down we got several suggestions:

1. Analysis Summary (you guessed it)

2. Timeline View

    CPU activity

    Threads -- each CPU has a significant colour
    All should be the same colour??



3. Diagnostics Summary

4. Symbol Resolution Logs

5. Files

Files generated


### Nsight compute


### Nsight graphics


## Memory Efficiently
To fully understand how to use memory 
efficently its important to understand 
CUDA memory hierarcy.

## Cuda memory Hierarchy 3.3 and 3.4

    [ Global Memory ] High latency
          |
      [L2 Cache] Medium latency
      /        \
      SM        SM Lower Latency
  [Texture]     [Texture]
  [Constant]    [Constant]
  [Shared]      [Shared]
  [Registers]   [Registers] Lowest Latency

CudaMalloc allocates in Global Memory
L2 cache cpl mb
Thread-level parallelism
Coalescing memory

### Coalescing memory 

The word coalesced means that small parts come together as a whole. 
Like combining or merging.
The term is used in Cuda to describe how memory is to be read by different
threads to make execution more efficent.
The opposite of coaleced memory read is strided memory read.

For a more intiuite explanation:
[coalesce memory](https://www.youtube.com/watch?v=mLxZyWOI340)

### What does it mean to coalesce memory?

When a GPU thread is reading a memory chunk (global, constant, shared) 
the instruction will read a fixed size chunk of memory.

However, the thread might only need a certain amount of the chunk.
So the read instruction will be inefficent.

However if the program can be written in a manner so that the 
read instruction is fully utilized, ie many threads use the same
read instruction these threads will utilze coalesced, merged,
memory.


### Some more thoughts on the art of coalescing memory.

So its expensive to read from global memory, piece by piece.
To mitgate that we want to read a do a coaleshed memory read,
coalesce the whole memory into one call. Making the calls for global memory less.

For the rgb (16) bit we do not want to call each r, g, b for each pixel we want to coalesche the whole pixel as one call.

This can be done with the __align__ command. Telling nvcc to se the whole struct as a complete call.

If we want memory to coalcesce we need the momory not just to be adjecent.
They need to be aligned on a 32, 64 or 128 byte size.

So the image need to be a multiple of these. Since some warps will not be full, alot of warps will 
have to call on the GlobalMemory, creating alot of unecessary calls.

To handle these problems it can be better to work with padded arrays.
Drawing down on the memory calls.

Allocate -> copy to padded 2d buffer -> copy back to host


### Texture and Constant Memory

__constant__  1  Read only for kernels.
              2. cached on SM  in Constant cache
              3. Fast when all SM's read the same address in the constant cache,
              it will be handled as a single request and will be very fast.
              4. Optimized for 2D locality

But : If Threads reads from different adresses, seperate requests, this will slow down performance.

3.4 Problem -- We want to pass alot of data from device to the kernel.
    1. CUDA can only pass 4kB to kernel directly, i.e no CudaMalloc
    2. Pass it to global memory, i.e do CudaMalloc
     2.a This will be ineffective, all Threads are accessing global memory making different request

    Solution:
    Adding the __constant__ keyword to get around this

    1. instance of object with __constant__ keyword
    2. Create on host
    3. Copy with cudaMemcpyToSymbol -> copies data into constant memory
    4. This changes the request to be the same for each SM, speeding up the process.

Texture API : 1. Handles filtering, bounce checking and so on
              2. Some boiler plate to set up
								Makes host code more verbose but 
								the device code will get smaller.
              3. Data should be copied to CUDA arrays

3.4 Problem : accessing alot of pixel data might be expensive if working with 2D data 

		
    The boiler plate to use Texture API:
        -> use cudaArray : buffers for optimized texture formulation

        resourceDescriptions : how to handle read outside and interpolation
        textureDescriptions


        Creating an Array With a float buffer as :
        cudaMemcpy2DToArray 
        dst  - Destination memory address
        wOffset - Destination starting X offset
        hOffset - Destination starting Y offset
        src - Source memory address
        spitch - Pitch of source memory
        width - Width of matrix transfer (columns in bytes)
        height - Height of matrix transfer (rows)
        kind - Type of transfer

        This places the memory in the Texture Memory and optimizes.

			Note: that this is very similar to the OpenGl textures!



## Instructions and Control Flow Optimaztion 3.5

The first thing to handle is really to see that memory movement is handled efficently.
There are some teqniques to further improve performance.

Use Integral Types:
    float4 uint2 of in vector format (as vec4 in GLSL)
    -> Generates wider memory transactions
    -> handled well by the compiler.
    float4 instead of allign as eg.

SP's can handle instruction level parallellism:
    I.e SMs can run multiple independent instructions at once on one SP at one time
    This can hide memory latency (400+ cycles)
    Can hide arithmetic latency (around 20 cycles)
    So if we want to really max out we want more instructions running at once.

    This can be done by computing multiple independent results per thread.
    By adding more outputs in a thread this can be achieved.

    Caveats:
        Entire warp must run the same instructions
        If and loop statemenst can cause divergence
        Some threads are left idle while others run

    Also to notice:
        Fast, low precision versions for floating point functions
        Division, trigometry, log, exp
        All are named with leading __
        Enable everywhere with --use_fast_math

# Section 4 : Paralell Algorithms 

1. Introduction to shared memory.
2. Reduction
3. Prefix Sum
4. Filtering

## Section 4.1 Shared Memory : Transposing a Matrix

 Matrix transpose, does not fit as well in the CUDA modell, locality of output makes the call some what difficult.

This brings us on to shared memory. This maximizes coaleshed memeory when working with read/write to different locations.

__shared__ to utilize shared memory within blocks within the SM.

### cooperative\_groups::thread\_block 

Can be seen as a kind of join, that waits for all the threads to finish within a block 
before continuing. This becomes necessary when we need to wait until the shared memory 
has been filled until reading from it.


See above for the shared memory location within the SM.

    64-96kB of shared memory
    Read/write access from kernels (so have to be handled in kernel code)
    Shared with a thread block: so can share data between threads

    __shared__ prefix siginifies that the data is shared inbetween threads. 
    One can se it as each block gets a shared instance of this element.

    Caveats:

    Shared memory is not infinite. So it must not be exeeded per warp, so it might effect occupancy.

    Bank Conflicts:
        The shared memory is organized into 32 banks  

 Bank     |      1      |      2      |      3      |...
 Addresses|  0  1  2  3 |  4  5  6  7 |  8  9 10 11 |...
 Addresses| 64 65 66 67 | 68 69 70 71 | 72 73 74 75 |           
    .            .             .             .

        Each bank can only access one location( = float = 4 bytes) at the time.
        Thread 0 reads first element of shared tile. 0x0
        Thread 1 reads from the 32nd element of shared tile. 0x64
        Here the elements are the locations, so they will try to access the same bank, 
        and one of the threads will have to wait for the first to finish.
        However if they read from the same exact same address, 0x0, the bank can broadcast this to those threads.

        Read the CUDA examples on transpose to understand the exact perks of optimazation.
        One seems to be to make the stride larger than actually needed by adding a columne.


## Section 4.2 Reduction

A basic building block for many parallell algorithms, makes understand how to communicate inbetween threads.

Reduction : Combine all elements of an array to produce a single value, binary or function.

### Reduce an array with +operator
A nice log\_2 n instead of n

Caveats:
    1. More data elements than processors
    2. Memory access patterns

    Shared memory can be used within blocks,
    but sharing between blocks is tricky.

		So first we can read one block into shared memory

Possible improvments:

1. Section reduction uses a single thread
2. Shared memory bank conflicts
3. Fine-tuning

For more finetuning improvments check out CUDA Examples:
[reduction](~/NVIDIA_CUDA-11.5_Samples/6_Advanced/reduction)


## FAQ


Q1: What is the differenve between \_\_threadfence() and \_\_syncthreads().

These things are quite different, it all comes down to synchronization of blocks
and threads when running kernel. 
Threadfence belongs to the fencing funcions and are commonly used when
threads consume some data that has been produced by other threads.
Synchtreads belongs to the synchronization functions.

1. \_\_threadfence() 
	Is used to mitigate a problem.
	The porblem is that there is no guratee that a block will 
	know that another block writes to something to global memory.
	There is no guarantee for the order of writing to global memory
	outside of a block.

	\_\_threadfence function stalls current thread until its writes to global memory are guaranteed to be visible by all other threads in the grid. So, if you do something like:

	store your data
	\_\_threadfence()
	atomically mark a flag

2. \_\_syncthreads()
Used to coodinate communication between the threads of the same block.
However the new modern way to sync is to use [cooperative groups](https://developer.nvidia.com/blog/cooperative-groups/)

Q2: What are atomic operations?

From the programming guide:
``` 
An atomic function performs a read-modify-write atomic operation on one 32-bit or 64-bit word residing in global or shared memory. 
For example, atomicAdd() reads a word at some address in global or shared memory, adds a number to it, and writes the result back to the same address.
The operation is atomic in the sense that it is guaranteed to be performed without interference from other thread
```


    






    
 





