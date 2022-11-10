#include <GLFW/glfw3.h>
#include <GL/glew.h>

#include "glfw_state/glfw_state.hpp"
#include "imgui/imguiBackend.hpp"
#include "opengl/gl.hpp"

int main(int argc, char *argv[])
{

  GLFW_State_t * backend_state = BackendSetup(
      "cuda test"
      512, 512
      );

  GlSetup();

  const char* glsl_version ="#version 130";

  UiBackend ui;

	std::shared_ptr<bool> quit(new bool(false));

  ui.addElementContainer("R", new SideBar("side bar", 200, 400));
	ui.addElement("R", "Quit", new Button(quit));

	ui.addElementContainer("W", new Window("window", 20, 20));

  ui.linkGlfwGL3(backend_state->window, glsl_version);

  while(true){

    if(glfwWindowShouldClose(bs->window) || *quit.get() ){
      break;
    }

    ui.draw();

		glfwSwapBuffers(bs->window);
		glfwPollEvents();

  }

  return EXIT_SUCCESS;

}
