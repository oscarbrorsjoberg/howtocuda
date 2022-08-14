// Utility functions for example programs.

#include <assert.h>
#include <getopt.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
/* #include <cuda_runtime_api.h> */
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include "utils.h"

void printCudaInformation()
{

	/* int darabszam; */
	int driverVersion = 0, runtimeVersion = 0;
	int devCount;


	cudaGetDeviceCount(&devCount);
	if(devCount == 0){
		std::cout << "No supported CUDA device found\n";
	}
	else{
		std::cout << devCount<< " CUDA device(s) found\n";
	}

	for(int dev = 0; dev < devCount; ++dev){
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		std::cout << dev << " device name " 
			<< deviceProp.name << "\n";

		std::cout << "CUDA capability version "
			<< deviceProp.major << "." << deviceProp.minor << "\n";

		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);

		// Versions
		std::cout << "CUDA driver verison / Runtime version " <<
			driverVersion / 1000 << " " << (driverVersion % 100) / 10
			<< "/" <<
			runtimeVersion / 1000 << " " << (driverVersion % 100) / 10 <<
			"\n";

		// VRAM
		std::cout << "Total VRAM: " << static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f) << " MBytes\n"
			<< "number of bytes: "  << (unsigned long long)(deviceProp.totalGlobalMem) << "\n";

		// Cores
		std::cout << deviceProp.multiProcessorCount << " Muliprocessores,\n" 
			 << _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) << " CUDA cores \n"
			 << "MP: " << _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)
			     * deviceProp.multiProcessorCount << " CUDA cores\n";

		// Clock rate
		std::cout << "GPU max clock: \t" << deviceProp.clockRate * 1e-3f << "MHz "
			<< '(' << deviceProp.clockRate * 1e-6f << "GHz)\n";

		// VRAM info
		std::cout <<
			"VRAM clock " << deviceProp.memoryClockRate * 1e-3f << "MHz\n";
		std::cout <<
			"VRAM transfer rate: " << deviceProp.memoryBusWidth << "-bit\n";

		// Threads warps and stuff
		std::cout <<
			"Warp size: " << deviceProp.warpSize << "\n";
	std::cout <<
			"Max threads / Multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << "\n";
	std::cout <<
			"Max threads / Block: " << deviceProp.maxThreadsPerBlock << "\n";
	std::cout <<
			"Max Block size (x,y,z):  ("
			<< deviceProp.maxThreadsDim[0] << "," 
			<< deviceProp.maxThreadsDim[1] << "," 
			<< deviceProp.maxThreadsDim[2] << ")\n";
	std::cout <<
			"Max Grid dim (x,y,z):  ("
			<< deviceProp.maxGridSize[0] << "," 
			<< deviceProp.maxGridSize[1] << "," 
			<< deviceProp.maxGridSize[2] << ")\n";
	}

}


