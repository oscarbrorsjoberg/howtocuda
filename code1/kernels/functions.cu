/******************************************************************************
* File:             functions.cu
*
* Author:           Oscar Sjoeberg  
* Created:          01/17/22 
* Description:			Some Cuda Functions
*****************************************************************************/


#include <assert.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

// Standard CUDA API functions
#include <cuda_runtime_api.h>
// CUDA cooperative groups API
#include <cooperative_groups.h>

#include "KernelTimer.hpp"
#include "functions.hpp"
#include "utils.h"


__global__ void add_kernel(float *dest, int n_elts, 
													const float *a, const float *b)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index >= n_elts) return;
	dest[index] = a[index] + b[index];
}

std::vector<float> CUaddVectors(const std::vector<float> &a, const std::vector<float> &b)
{
	assert(a.size() == b.size());

	std::vector<float> out(a.size());
	float *dest = out.data();

	float *devArrayA, *devArrayB, *devArrayDest;
	int arraySize = (size_t)(sizeof(float)*a.size());
	// Create Device Arrays
	ck(cudaMalloc(&devArrayA,    arraySize));
	ck(cudaMalloc(&devArrayB,    arraySize));
	ck(cudaMalloc(&devArrayDest, arraySize));

	// Transfer to device
	ck(cudaMemcpy(devArrayA, a.data(), arraySize, cudaMemcpyHostToDevice));
	ck(cudaMemcpy(devArrayB, b.data(), arraySize, cudaMemcpyHostToDevice));

	const int BLOCK_SIZE = 128;
	int nBlocks = (a.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
	// running kernel
	{
		KernelTimer t;
		t.start();
		add_kernel<<<BLOCK_SIZE, nBlocks>>>(devArrayDest, a.size(),
																			devArrayA, devArrayB);
		t.stop();
		ck(cudaMemcpy(dest, devArrayDest, arraySize,
					cudaMemcpyDeviceToHost));
	}

	return out;

}
