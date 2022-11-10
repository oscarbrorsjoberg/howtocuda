// summing an array in cuda

#include <assert.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>

// Standard CUDA API functions
#include <cuda_runtime_api.h>
// CUDA cooperative groups API
#include <cooperative_groups.h>

#include "utils.h"


__device__ unsigned int blocks_finished = 0;

__device__ bool wait_for_all_blocks()
{
	// wait until global wirte is visible to all other blocks
	__threadfence();
  // Wait for all blocks to finish by atomically incrementing a counter
	// since atomic the counter is shared
	bool is_last = false;
  if (threadIdx.x == 0) {
    unsigned int ticket = atomicInc(&blocks_finished, gridDim.x);
    is_last = (ticket == gridDim.x - 1);
  }
  if (is_last) {
    blocks_finished = 0;
  }
  return is_last;
}


__device__ int reduce_block(const int *source, int shared_data[], 
		cooperative_groups::thread_block block)
{
	// select all even block indexes
	unsigned int index = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	auto tid = threadIdx.x;

	// adding the even block with one item away
	// if block dim is 10 for the threads
	// i.e 0, -> 10
	// i.e 1, -> 11 
	shared_data[tid] = source[index] + source[index + blockDim.x];

	cooperative_groups::sync(block);

	for(int stride = 1; stride < blockDim.x; stride *=2 ){
		int index = 2 * stride * tid;
		if(index < blockDim.x){
			shared_data[index] += shared_data[index + stride];
		}
		cooperative_groups::sync(block);
	}
	return shared_data[0];
}

__global__ void reduce(const int *source, int *dest)
{
	// can set the shared data size at runtime
	// so no need for fixed size in device code
	extern __shared__ int shared_data[];


	int block_result =
		reduce_block(source, shared_data, cooperative_groups::this_thread_block());

	// First thread of each block writes the block result into global memory
	if(threadIdx.x == 0){
		dest[blockIdx.x] = block_result;
	}
	bool is_last = wait_for_all_blocks();


	if(is_last){
		// do these writes in parallell
		/* unsigned int index = 2 * blockIdx.x * blockDim.x + threadIdx.x; */

		for(int stride = 1; stride < gridDim.x; stride *=2 ){
			int index = 2 * stride * threadIdx.x;
			if(index < gridDim.x){
				dest[index] += dest[index + stride];
			}
			cooperative_groups::sync(cooperative_groups::this_thread_block());
		}
	}
}

int main(int argc, char **argv)
{
	const unsigned int COUNT = 4096 * 4096;
	std::unique_ptr<int[]> source(new int[COUNT]);

	std::mt19937 rng;
	rng.seed(0);
	std::uniform_int_distribution<std::mt19937::result_type> dist(0,9);

	for (int i = 0; i < COUNT; ++i) {
		source[i] = dist(rng);
	}

	int *source_dev, *dest_dev;
	size_t size = COUNT * sizeof(int);

	ck(cudaMalloc(&source_dev, size));
	ck(cudaMemcpy(source_dev, source.get(), size, cudaMemcpyHostToDevice));

	int BLOCK_SIZE = 128;
	int n_blocks = (COUNT + BLOCK_SIZE - 1) / (2 * BLOCK_SIZE);
	ck(cudaMalloc(&dest_dev, n_blocks * sizeof(int)));

	
	int result;
	{
		KernelTimer t;
		size_t shared_memory_size = BLOCK_SIZE * sizeof(int);
		t.start();
		reduce<<<n_blocks, BLOCK_SIZE, shared_memory_size>>>(source_dev, dest_dev);
		t.stop();


		ck(cudaMemcpy(&result, dest_dev, 
          sizeof(result), cudaMemcpyDeviceToHost));

	}

	ck(cudaFree(source_dev));
	ck(cudaFree(dest_dev));

	int result_reference = std::accumulate(source.get(), source.get() + COUNT, 0);
	std::cout << "Sum of " << COUNT << " elements: " << result << "\n";
	assert(result_reference = result);

	return 0;
}
