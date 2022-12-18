#include <iostream>
#include <filesystem>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "glfw_state.hpp"
#include "imguiBackend.hpp"
#include "gl.hpp"
#include "ims/ims.hpp"
#include "utils.h"

/* __global__ { */
/* } */

namespace fs = std::filesystem;


static GLuint *create_texture_id(
    unsigned int width,
    unsigned int height)
{
  GLuint *out = new GLuint;
  glGenTextures(1, out);
  glBindTexture(GL_TEXTURE_2D, *out);

  // set basic parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, 
              height, 0, GL_RED, GL_FLOAT, NULL);


  return out;
}




int main(int argc, char *argv[])
{

  if(argc < 2){
    std::cout << "Usage ui_window <image>\n";
    return EXIT_FAILURE;
  }

  
  std::string in_path(argv[1]);
  cudaGraphicsResource *cuda_tex_screen_resource;

  GLFW_State_t * bs = BackendSetup(
      "cuda test",
      512, 512
      );

  GlSetup();

  planar_image_t image_in;


  if(!CU_readppm_planar_image(in_path, image_in)){
    std::cerr << "Unable to read image " << in_path << "\n";
    planar_image_free(image_in);
    return EXIT_FAILURE;
  }

  GLuint *texId = create_texture_id(image_in.width, 
                                    image_in.height);

  if(!glIsTexture(*texId)){
    std::cout << "Unable to set up texture\n";
    return EXIT_FAILURE;
  }

  ck(cudaGraphicsGLRegisterImage(&cuda_tex_screen_resource, *texId, GL_TEXTURE_2D,
      cudaGraphicsMapFlagsReadOnly));



  cudaArray_t red_channel;

  std::cout << "number of bytes " << sizeof(float) * image_in.width << "\n";

  ck(cudaGraphicsMapResources(1,
                  &cuda_tex_screen_resource, 0));

  ck(cudaGraphicsSubResourceGetMappedArray(
                  &red_channel, 
                  cuda_tex_screen_resource,
                  0, 0
                  ));

  ck(cudaMemcpy2DToArray(red_channel, 0, 0, (void*)image_in.r, 4*image_in.width, 4*image_in.width, image_in.height, cudaMemcpyDeviceToDevice));


  ck(cudaGraphicsUnmapResources(1, 
                  &cuda_tex_screen_resource, 0));


  const char* glsl_version ="#version 130";

  UiBackend ui;

	std::shared_ptr<bool> quit(new bool(false));

  ui.addElementContainer("R", new SideBar("side bar", 200, 400));
	ui.addElement("R", "Quit", new Button(quit));
	ui.addElementContainer("W", new Window("window", 20, 20));

	ui.addImage("W", texId, image_in.width, image_in.height); 

  ui.linkGlfwGL3(bs->window, glsl_version);

  while(true){
    glClearColor(0.0f, 0.0f, 0.0f ,1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if(glfwWindowShouldClose(bs->window) || *quit.get() ){
      break;
    }

    ui.draw();

		glfwSwapBuffers(bs->window);
		glfwPollEvents();

  }

  delete texId;

  return EXIT_SUCCESS;

}
