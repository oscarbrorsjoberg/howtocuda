#ifndef UTILS_H_KKV2E5DQ
#define UTILS_H_KKV2E5DQ

#include <cuda_runtime_api.h>

// cuda error interpreter
#define ck(code) 		                     \
{																         \
	if((code) != cudaSuccess) {		         \
		fprintf(stderr,											 \
		"Cuda failed due to %s:%d: '%s' \n", \
				__FILE__,												 \
				__LINE__,												 \
		cudaGetErrorString(code));				   \
	}																			 \
}																				 \


void printCudaInformation();



#endif /* end of include guard: UTILS_H_KKV2E5DQ */
