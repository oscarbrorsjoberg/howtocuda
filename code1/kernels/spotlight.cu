#include <cmath>
#include <cstdint>
#include <iostream>
#include <fstream>

#include <string>
#include <cuda_runtime_api.h>

#include "utils.h"
#include "ims/ims.hpp"

struct light {
  float x;
  float y;
  float radius;
  float brightness;
};

// spotlight stuff

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

  /* light light1 = {0.2, 0.1, 0.1, 4.0}; */
  /* light light2 = {0.25, 0.2, 0.075, 2.0}; */
  /* light light3 = {0.5, 0.5, 0.3, 0.3}; */
  /* light light4 = {0.7, 0.65, 0.15, 0.8}; */

  // these will be loaded on device
  /* pixel_t *input_image = nullptr; */
  /* int input_im_h = input_im_w = 0; */

  /* pixel_t *output_image = nullptr; */
  /* int output_im_h = output_im_w = 0; */

  planar_image_t image_in;
  if(!CU_readppm_planar_image(in_path, image_in)){
    std::cerr << "Unable to read image " << in_path << "\n";
    planar_image_free(image_in);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}


