#include <cuda_runtime_api.h>

/* A single pixel with floating-point channel values */
struct __align__ (16) pixel_t {
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
  int width;
  int height;
};

/* bool loadPPM(const char *file, pixel_t **data, unsigned int *w, unsigned int *h); */
/* void savePPM(const char *file, pixel_t *data, unsigned int w, unsigned int h); */

bool CU_readppm_planar_image(const std::string &input_path, planar_image_t &image);
planar_image_t planar_image_create(int width, int height);
void planar_image_free(planar_image_t &img);
