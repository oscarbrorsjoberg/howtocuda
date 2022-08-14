#include <cmath>
#include <cstdint>

#include <string>
#include <cuda_runtime_api.h>

#include "utils.h"

struct light {
  float x;
  float y;
  float radius;
  float brightness;
};


int main(int argc, char *argv[]){

  std::string in, out;
  if(argc == 2){
    in(argv[1]);
    out(argv[2]);
  }


  light light1 = {0.2, 0.1, 0.1, 4.0};
  light light2 = {0.25, 0.2, 0.075, 2.0};
  light light3 = {0.5, 0.5, 0.3, 0.3};
  light light4 = {0.7, 0.65, 0.15, 0.8};

  // these will be loaded on device
  pixel *input_image = nullptr;
  int input_im_h = input_im_w = 0;
  pixel *output_image = nullptr;
  int output_im_h = output_im_w = 0;

  if (!loadPPM(current, &host_image, &params.width, &params.height)) {
    std::cerr << "Unable to load data " << in << "\n";
  }
  image_size 
    


  free_image(input2d);
  free_image(output2d);

  return EXIT_SUCCESS;
}


