#include <assert.h>
#include <getopt.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <cuda_runtime_api.h>

/* #include <helper_cuda.h> */
#include "KernelTimer.hpp"

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
}
