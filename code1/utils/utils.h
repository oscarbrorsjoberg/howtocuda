#ifndef UTILS_H_KKV2E5DQ
#define UTILS_H_KKV2E5DQ


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

/* A single pixel with floating-point channel values */
struct __align__ (16) pixel {
  float red;
  float green;
  float blue;
  float alpha;
};

/* An image with planar layout: separate buffers for each color channel */
struct image {
  float *red;
  float *green;
  float *blue;
};

bool loadPPM(const char *file, pixel **data, unsigned int *w, unsigned int *h);
void savePPM(const char *file, pixel *data, unsigned int w, unsigned int h);


#endif /* end of include guard: UTILS_H_KKV2E5DQ */
