#include <__clang_cuda_builtin_vars.h>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime_api.h>

#include "utils.h"
#include "ims.hpp"


/*
  @ Converts rgb to monochrome bad alignement
 */
__global__ void bad_monochrome(const upixel_t *source, upixel_t *dest, int size)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size) return;

  float grey(
      source[index].r * 0.3125f +
      source[index].g * 0.5f +
      source[index].r * 0.1875f
      );

  dest[index].r = grey;
  dest[index].g = grey;
  dest[index].b = grey;
  dest[index].a = source[index].a;
}

/*
  @ Converts rgb to monochrome bad alignement
 */
__global__ void monochrome(const pixel_t *source, pixel_t *dest, int size)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= size) return;

  float grey(
      source[index].r * 0.3125f +
      source[index].g * 0.5f +
      source[index].r * 0.1875f
      );

  dest[index].r = grey;
  dest[index].g = grey;
  dest[index].b = grey;
  dest[index].a = source[index].a;
}


int main (int argc, char *argv[])
{
  if(argc != 3){
    std::cout << "Usage monochrome <image_rgb> <image_grey>\n";
    return EXIT_FAILURE;
  }
  else{
    return EXIT_SUCCESS;
  }
}
