#include <assert.h>
#include <getopt.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <cuda_runtime_api.h>

#include "ims.hpp"

/******************************************************************************
* File:             ppm.cu
*
* Author:             
* Created:          04/14/22 
* Description:      A ppm read write for cuda alignment
*****************************************************************************/

__global__ void unpack_image(image planar, const pixel *packed, int pixel_count)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pixel_count) return;

  planar.red[index] = packed[index].red;
  planar.green[index] = packed[index].green;
  planar.blue[index] = packed[index].blue;
}

__global__ void pack_image(const image planar, pixel *packed, int pixel_count)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pixel_count) return;

  packed[index].red = planar.red[index];
  packed[index].green = planar.green[index];
  packed[index].blue = planar.blue[index];
}


static const unsigned int HEADER_SIZE = 0x40;
static const unsigned int CHANNELS = 3;

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
    pixel_data[i].red = raw_data[3 * i] * scale;
    pixel_data[i].green = raw_data[3 * i + 1] * scale;
    pixel_data[i].blue = raw_data[3 * i + 2] * scale;
  }

  *data = pixel_data;
  free(raw_data);

  return true;
}

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

image malloc_image(int pixel_count)
{
  image result;
  cudaCheckError(cudaMalloc(&result.red, pixel_count * sizeof(float)));
  cudaCheckError(cudaMalloc(&result.green, pixel_count * sizeof(float)));
  cudaCheckError(cudaMalloc(&result.blue, pixel_count * sizeof(float)));

  return result;
}

void free_image(const image &img)
{
  cudaCheckError(cudaFree(img.red));
  cudaCheckError(cudaFree(img.green));
  cudaCheckError(cudaFree(img.blue));
}

bool CU_getppm(const std::string &input_path, image *im)
{
	int width, height;

	pixel *host_image = nullptr;
	if (!loadPPM(input_path.c_str(), 
				&host_image, &width, &height)) {
		std::cerr << "Couldn't read image " << input_path << "\n";
		return false;
	}

	// allocating planar image
	ck(cudaMalloc());
	ck(cudaMalloc());
	ck(cudaMalloc());



}
