#ifndef UTILS_H_KKV2E5DQ
#define UTILS_H_KKV2E5DQ


#include <cuda_runtime_api.h>

#define ck(code) 		         \
{																         \
	if((code) != cudaSuccess) {		         \
		fprintf(stderr,											 \
		"Cuda failed due to %s:%d: '%s' \n", \
				__FILE__,												 \
				__LINE__,												 \
		cudaGetErrorString(code));				   \
	}																			 \
}																				 \

class KernelTimer
{

 public:
  KernelTimer();
  ~KernelTimer();

	void start();
	void stop();

 private:
	cudaEvent_t start_;
	cudaEvent_t stop_;

};


#endif /* end of include guard: UTILS_H_KKV2E5DQ */
