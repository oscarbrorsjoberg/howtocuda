#include <assert.h>
#include <getopt.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <cuda_runtime_api.h>

#include "utils.h"
#include "ims.hpp"

/******************************************************************************
* File:             ppm.cu
*
* Author:             
* Created:          04/14/22 
* Description:      A ppm read write for cuda alignment
*****************************************************************************/

__global__ void unpack_image(planar_image_t planar, const pixel *packed, int pixel_count)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pixel_count) return;

  planar.red[index] = packed[index].red;
  planar.green[index] = packed[index].green;
  planar.blue[index] = packed[index].blue;
}

__global__ void pack_image(const planar_image_t planar, pixel *packed, int pixel_count)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pixel_count) return;

  packed[index].red = planar.red[index];
  packed[index].green = planar.green[index];
  packed[index].blue = planar.blue[index];
}


static const unsigned int HEADER_SIZE = 0x40;
static const unsigned int CHANNELS = 3;

/******************************************************************************
* Function:         
* Description:      
* Where:
* Return:           
* Error:            
*****************************************************************************/

bool loadPPM(const char *file, pixel **data, unsigned int *w, unsigned int *h)
{
  FILE *fp = fopen(file, "rb");

  if (!fp) {
    std::cerr << "loadPPM() : failed to open file: " << file << "\n";
    return false;
  }

  // check header
  char header[HEADER_SIZE];

  if (fgets(header, HEADER_SIZE, fp) == nullptr) {
    std::cerr << "loadPPM(): reading header returned NULL\n";
    return false;
  }

  if (strncmp(header, "P6", 2)) {
    std::cerr << "unsupported image format\n";
    return false;
  }

  // parse header, read maxval, width and height
  unsigned int width = 0;
  unsigned int height = 0;
  unsigned int maxval = 0;
  unsigned int i = 0;

  while (i < 3) {
    if (fgets(header, HEADER_SIZE, fp) == NULL) {
      std::cerr << "loadPPM() : reading PPM header returned NULL" << std::endl;
      return false;
    }

    if (header[0] == '#') {
      continue;
    }

    if (i == 0) {
      i += sscanf(header, "%u %u %u", &width, &height, &maxval);
    } else if (i == 1) {
      i += sscanf(header, "%u %u", &height, &maxval);
    } else if (i == 2) {
      i += sscanf(header, "%u", &maxval);
    }
  }

  size_t pixel_count = width * height;
  size_t data_size = sizeof(unsigned char) * pixel_count * CHANNELS;
  unsigned char *raw_data = static_cast<unsigned char *>(malloc(data_size));
  *w = width;
  *h = height;

  // read and close file
  if (fread(raw_data, sizeof(unsigned char), pixel_count * CHANNELS, fp) == 0) {
    std::cerr << "loadPPM() read data returned error.\n";
  }
  fclose(fp);

  pixel *pixel_data = static_cast<pixel *>(malloc(pixel_count * sizeof(pixel)));
  float scale = 1.0f / 255.0f;
  for (int i = 0; i < pixel_count; i++) {
    pixel_data[i].r = raw_data[3 * i + 0] * scale;
    pixel_data[i].g = raw_data[3 * i + 1] * scale;
    pixel_data[i].b = raw_data[3 * i + 2] * scale;
  }

  *data = pixel_data;
  free(raw_data);

  return true;
}
/******************************************************************************
* Function:         
* Description:      
* Where:
* Return:           
* Error:            
*****************************************************************************/
void savePPM(const char *file, pixel *data, unsigned int w, unsigned int h)
{
  assert(data != nullptr);
  assert(w > 0);
  assert(h > 0);

  std::fstream fh(file, std::fstream::out | std::fstream::binary);

  if (fh.bad()) {
    std::cerr << "savePPM() : open failed.\n";
    return;
  }

  fh << "P6\n";
  fh << w << "\n" << h << "\n" << 0xff << "\n";

  unsigned int pixel_count = w * h;
  for (unsigned int i = 0; (i < pixel_count) && fh.good(); ++i) {
    fh << static_cast<unsigned char>(data[i].red * 255);
    fh << static_cast<unsigned char>(data[i].green * 255);
    fh << static_cast<unsigned char>(data[i].blue * 255);
  }

  fh.flush();

  if (fh.bad()) {
    std::cerr << "savePPM() : writing data failed.\n";
    return;
  }

  fh.close();
}

planar_image_t planar_image_create(int pixel_count)
{
  image out;
  ck(cudaMalloc(&result.red, pixel_count * sizeof(float)));
  ck(cudaMalloc(&result.green, pixel_count * sizeof(float)));
  ck(cudaMalloc(&result.blue, pixel_count * sizeof(float)));
  return result;
}

void planar_image_free(const planar_image &img)
{
  ck(cudaFree(img.r));
  ck(cudaFree(img.g));
  ck(cudaFree(img.b));
}

// this is stolen from corse, why 128?
constexpr int BLOCK_SIZE = 128;

bool CU_readppm(const std::string &input_path, planar_image_t *device_image)
{
	int width, height;
	pixel *host_pixels = nullptr;
	if (!loadPPM(input_path.c_str(), 
				&host_pixels, &width, &height)) {
		std::cerr << "Couldn't read image " << input_path << "\n";
	}



  int pixel_count = width * height;
  *device_image = planar_image_create(pixel_count);

  size_t image_size = width * height * sizeof(pixel);
  cudaCheckError(cudaMemcpy(params.input_image, host_image, image_size,
        cudaMemcpyHostToDevice));

  // number of pixels per block?
  int number_blocks = (pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE;

  unpack_image<<<number_blocks, BLOCK_SIZE>>>(*device_image, host_pixels,
                                              pixel_count);
 
  ck(CudaFree())

}



