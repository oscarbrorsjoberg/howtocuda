#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>

#include <string>
#include <cuda_runtime_api.h>

#include "utils.h"
#include "KernelTimer.hpp"
#include "ims/ims.hpp"

struct light_t {
  float x;
  float y;
  float radius;
  float brightness;
};

// spotlight stuff

__device__ float clamp(float value) { return value > 1.0f ? 1.0f : value; }

__device__ float light_brightness(float x, float y, unsigned int width, 
                                  unsigned height, const light_t &light)
{

  float norm_x = x / width;
  float norm_y = y / height;

  float dx = norm_x - light.x;
  float dy = norm_y - light.y;
  float dsqrd = dx * dx + dy * dy;
  if(dsqrd > light.radius * light.radius){
    return 0;
  }

  float distance = sqrtf(dsqrd);

  float scaled_distance = distance / light.radius;

  if(scaled_distance > .8){
    return (1.0f - (scaled_distance - 0.8f * 5.0f) * light.brightness );
  }
  else{
    return light.brightness;
  }
}

__global__ void spotlights(const planar_image_t source, 
                           planar_image_t dest, 
                           float ambient, 
                           light_t light_1,
                           light_t light_2,
                           light_t light_3,
                           light_t light_4
                           )
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;


  if(x >= source.width || y >= source.height){ 
    return;
  }

  int index = y * source.width + x;

  float brightness = ambient;

  /* for(int i = 0; i < number_of_lights; ++i){ */
  brightness += light_brightness(x,y, source.width, source.height, light_1);
  brightness += light_brightness(x,y, source.width, source.height, light_2);
  brightness += light_brightness(x,y, source.width, source.height, light_3);
  brightness += light_brightness(x,y, source.width, source.height, light_4);
  /* } */

  dest.r[index] = clamp(source.r[index] * brightness);
  dest.g[index] = clamp(source.g[index] * brightness);
  dest.b[index] = clamp(source.b[index] * brightness);

}



int main(int argc, char *argv[]){

  std::string in_path, out_path;

  if(argc == 3){
    in_path = argv[1];
    out_path = argv[2];
  }
  else{
    std::cout << "Usage : spoitlight <in.ppm> <out.ppm> \n";
    return EXIT_FAILURE;
  }

  light_t light1 = {0.2, 0.1, 0.1, 4.0};
  light_t light2 = {0.25, 0.2, 0.075, 2.0};
  light_t light3 = {0.5, 0.5, 0.3, 0.3};
  light_t light4 = {0.7, 0.65, 0.15, 0.8};

  float ambient_brtness = 0.4f;

  planar_image_t image_in, image_out;
  if(!CU_readppm_planar_image(in_path, image_in)){
    std::cerr << "Unable to read image " << in_path << "\n";
    planar_image_free(image_in);
    return EXIT_FAILURE;
  }

  std::cout << "Read image w("<< image_in.width << ") h(" << image_in.height << ")\n";

  image_out = planar_image_create(image_in.width, image_in.height);

  // set up grid and block
  dim3 BLOCK_DIM(32,16);
  dim3 grid_dim((image_in.width + BLOCK_DIM.x - 1) / BLOCK_DIM.x,
                (image_in.height + BLOCK_DIM.y - 1) / BLOCK_DIM.y);

  
  {
  KernelTimer t;
  t.start();
  spotlights<<<grid_dim, BLOCK_DIM>>>(image_in, image_out,
                                      ambient_brtness,
                                      light1,
                                      light2,
                                      light3,
                                      light4);
  t.stop();
  }




  if(!CU_saveppm_planar_image(out_path, image_out)){
    std::cerr << "Unable to save image " << out_path << std::endl;
    return EXIT_FAILURE;
  }

  planar_image_free(image_in);
  planar_image_free(image_out);

  return EXIT_SUCCESS;
}


