
#include <cuda_runtime_api.h>

/* A single pixel with floating-point channel values */
struct __align__ (16) pixel {
  float r;
  float g;
  float b;
  float a;
};

// planar image
struct planar_image_t {
  float *r;
  float *g;
  float *b;
};

bool loadPPM(const char *file, pixel **data, unsigned int *w, unsigned int *h);
void savePPM(const char *file, pixel *data, unsigned int w, unsigned int h);

planar_image_t CU_readppm(const std::string &input_path);
planar_image_t planar_image_create(uint32_t number_of_pixels);
void planar_image_free(planar_image_t &image);
