#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "glfw_state.hpp"
#include "imguiBackend.hpp"
#include "gl.hpp"

__global__ {

}


int main(int argc, char *argv[])
{

  GLFW_State_t * bs = BackendSetup(
      "cuda test",
      512, 512
      );

  GlSetup();

  planar_image_t image_in, image_out;
  if(!CU_readppm_planar_image(in_path, image_in)){
    std::cerr << "Unable to read image " << in_path << "\n";
    planar_image_free(image_in);
    return EXIT_FAILURE;
  }

  const char* glsl_version ="#version 130";

  UiBackend ui;

	std::shared_ptr<bool> quit(new bool(false));

  ui.addElementContainer("R", new SideBar("side bar", 200, 400));
	ui.addElement("R", "Quit", new Button(quit));
	ui.addElementContainer("W", new Window("window", 20, 20));

	ui.addImage("W", , , ); 
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

  return EXIT_SUCCESS;

}
