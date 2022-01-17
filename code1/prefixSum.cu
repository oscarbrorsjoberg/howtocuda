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


// AKA SCAN



__device__ int block_scan(int idata, int shared_data[],
		cooperative_groups::thread_block block)
{


}




int main(int argc, char **argv){





	return 1;
}
