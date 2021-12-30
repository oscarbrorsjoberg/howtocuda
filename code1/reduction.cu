// summing an array in cuda

#include <assert.h>
#include <stdio.h>


#include <cuda_runtime_api.h>


#define checkCudaError(code) 		         \
{																         \
	if((code) != cudaSuccess) {		         \
		fprintf(stderr,											 \
		"Cuda failed due to %s:%d: '%s' \n", \
				__FILE__,												 \
				__LINE__,												 \
		cudaGetErrorString(code));				   \
	}																			 \
}																				 \



__device__ void reduce_block(const int *source, int shared_data[], 
		cooperative_groups::thread_block block)
{
	// index is one block away
	unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	auto tid = threadIdx.x;

	// adding index and in current block into shared memory 
	shared_data[tid] = source[index] + source[index + blockDim.x];

	cooperative_groups::sync(block);

	for(int stride = 1; stride < blockDim.x; stride *=2 ){

	}

}
