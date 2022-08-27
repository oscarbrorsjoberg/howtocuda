#include <assert.h>
#include <getopt.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <cuda_runtime_api.h>

#include "utils.h"
#include "ims/ims.hpp"

/******************************************************************************
* File:             ppm.cu
*
* Author:             
* Created:          04/14/22 
* Description:      A ppm read write for cuda alignment
*****************************************************************************/

__global__ void unpack_image(planar_image_t planar, const pixel_t *packed, int pixel_count)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pixel_count) return;

  planar.r[index] = packed[index].r;
  planar.g[index] = packed[index].g;
  planar.b[index] = packed[index].b;
}

__global__ void pack_image(const planar_image_t planar, pixel_t *packed, int pixel_count)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pixel_count) return;

  packed[index].r = planar.r[index];
  packed[index].g = planar.g[index];
  packed[index].b = planar.b[index];
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

static bool loadPPM(const char *file, pixel_t **data, unsigned int *w, unsigned int *h)
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

  pixel_t *pixel_data = static_cast<pixel_t*>(malloc(pixel_count * sizeof(pixel_t)));
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
static bool savePPM(const std::string &file, const pixel_t *pixels, 
                          int width, int height)
{
  assert(pixels != nullptr);
  assert(width > 0);
  assert(height > 0);

  std::fstream fh(file, std::fstream::out | std::fstream::binary);

  if (fh.bad()) {
    std::cerr << "savePPM() : open failed.\n";
    return false;
  }

  fh << "P6\n";
  fh << width << "\n" << height << "\n" << 0xff << "\n";

  unsigned int pixel_count = width * height;

  for (unsigned int i = 0; (i < pixel_count) && fh.good(); ++i) {
    fh << static_cast<unsigned char>(pixels[i].r* 255);
    fh << static_cast<unsigned char>(pixels[i].g * 255);
    fh << static_cast<unsigned char>(pixels[i].b * 255);
  }

  fh.flush();

  if (fh.bad()) {
    std::cerr << "savePPM() : writing data failed.\n";
    return false;
  }

  fh.close();
  return true;
}

planar_image_t planar_image_create(int width, int height)
{
  planar_image_t out;

  out.width = width;
  out.height = height;

 int pixel_count = width * height;

 ck(cudaMalloc(&out.r, pixel_count * sizeof(float)));
 ck(cudaMalloc(&out.g, pixel_count * sizeof(float)));
 ck(cudaMalloc(&out.b, pixel_count * sizeof(float)));

  return out;
}

void planar_image_free(planar_image_t &img)
{
  ck(cudaFree(img.r));
  ck(cudaFree(img.g));
  ck(cudaFree(img.b));
}

// this is stolen from corse, why 128?
constexpr int BLOCK_SIZE = 128;

/*
   reads host ppm and creates planar image on device from host
   */


bool CU_readppm_planar_image(
    const std::string &input_path, planar_image_t &device_image)
{
	unsigned int width, height;
  // loading pixels (as (rgb per pixel))
	pixel_t *host_pixels = nullptr;
	if (!loadPPM(input_path.c_str(),
				&host_pixels, &width, &height)) {
		std::cerr << "Couldn't read image " << input_path << "\n";
    return false;
	}
  if(host_pixels){
    int pixel_count = width * height;

    device_image = planar_image_create(width, height);
    size_t image_size = pixel_count * sizeof(pixel_t);
    pixel_t *device_pixels;

    ck(cudaMalloc(&device_pixels, (int)image_size));
    ck(cudaMemcpy(device_pixels, host_pixels, (int)image_size, cudaMemcpyHostToDevice));

    // number of pixels per block?
    int number_blocks = (pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // unpack image create planar image
    unpack_image<<<number_blocks, BLOCK_SIZE>>>(device_image, device_pixels,
        pixel_count);

    ck(cudaFree(device_pixels));
    free(host_pixels);
  }
  return true;
}

/*
  --   saves image as ppm (by packing it to pixels)
*/

bool CU_saveppm_planar_image(
    const std::string &output_path, const planar_image_t &device_image)
{

  int pixel_count = device_image.width * device_image.height;
  int number_blocks = (pixel_count + BLOCK_SIZE - 1) / BLOCK_SIZE;

  pixel_t *host_pixels;
  pixel_t *dev_pixels;

  int image_size = pixel_count * (int)sizeof(pixel_t);

  ck(cudaMalloc(&dev_pixels, image_size));

  host_pixels = new pixel_t[pixel_count];

  // unpack image create planar image
  pack_image<<<number_blocks, BLOCK_SIZE>>>(device_image, dev_pixels,
                                              pixel_count);
  ck(cudaMemcpy(host_pixels, dev_pixels, image_size, cudaMemcpyDeviceToHost));

  ck(cudaFree(dev_pixels));

  if(!savePPM(output_path, host_pixels, device_image.width, device_image.height)){
    std::cerr << "Unable to save image " << output_path << "\n";
    return false;
  }

  delete[] host_pixels;

  return true;
}

