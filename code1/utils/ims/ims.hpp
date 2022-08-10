

/* A single pixel with floating-point channel values */
struct __align__ (16) pixel {
  float red;
  float green;
  float blue;
  float alpha;
};

// planar image
struct image {

  float *red;
  float *green;
  float *blue;
};

bool loadPPM(const char *file, pixel **data, unsigned int *w, unsigned int *h);
void savePPM(const char *file, pixel *data, unsigned int w, unsigned int h);
