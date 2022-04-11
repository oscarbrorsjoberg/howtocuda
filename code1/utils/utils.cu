
// Utility functions for example programs.

#include <assert.h>
#include <getopt.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

#include "utils.h"



KernelTimer::KernelTimer()
{
	cudaEventCreate(&start_);
	cudaEventCreate(&stop_);
}

void KernelTimer::start(){
	cudaEventRecord(start_);
}

void KernelTimer::stop(){
	cudaEventRecord(stop_);
}

KernelTimer::~KernelTimer()
{
	float ms = 0.0;
	cudaEventSynchronize(stop_);
	cudaEventElapsedTime(&ms, start_, stop_);

	std::cout << "Kernel ran in " << ms << "ms \n";
	/* cudaEventDestroy(&start_); */
	/* cudaEventDestroy(&stop_); */
}

void printCudaInformation()
{

	int darabszam;
	int driverVersion = 0, runtimeVersion = 0;
	int devCount;


	cudaGetDeviceCount(&devCount);
	if(devCount == 0){
		std::cout << "No supported CUDA device found\n";
	}
	else{
		std::cout << devCount<< "CUDA device(s) found\n";
	}

	for(int dev = 0; dev < devCount; ++dev){
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudeGetDeviceProperties(&deviceProp, dev);

		std::cout << dev << "device name" 
			<< deviceProp.name << "\n";

		std::cout << "CUDA capability version" <<
			<< deviceProp.major << " " << deviceProp.minor << "\n";

		cudaDriverGetVersion(&driveVersion);
		cudaRuntimeGetVersion(&runtimeVersion);

		std::cout << "CUDA driver verison / Runtime version " <<
			driverVersion / 1000 << " " << (driverVersion % 100) / 10
			<< "/" <<
			runtimeVersion / 1000 << " " << (driverVersion % 100) / 10 <<
	}

}


