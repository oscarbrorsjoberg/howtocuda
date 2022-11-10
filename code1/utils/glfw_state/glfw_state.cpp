#include "glfw_state.hpp"

#include <iostream>
#include <cstring>

#include <GLFW/glfw3.h>

static void ErrorCallback(int error, const char* description)
{
	fprintf(stderr, "GLFW Error: [error=%d] %s\n", error, description);
}

static void KeyCallback(
		GLFWwindow* window,
		int key,
		int scancode,
		int action,
		int mods
		)
{
	(void)scancode;
	(void)mods;
	GLFW_State_t* state = (GLFW_State_t*)glfwGetWindowUserPointer(window);
	if (state->key != NULL && action == GLFW_PRESS)
		state->key(state, key);
}


static void CursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
	GLFW_State_t* state = (GLFW_State_t*)glfwGetWindowUserPointer(window);
	state->mouse_pos_prev_x = state->mouse_pos_x;
	state->mouse_pos_prev_y = state->mouse_pos_y;
	state->mouse_pos_x = xpos;

  // need to find what 
	state->mouse_pos_y = state->height - ypos;
	/* state->mouse_pos_y = ypos; */
}


static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	(void)mods;
	GLFW_State_t* state = (GLFW_State_t*)glfwGetWindowUserPointer(window);
	if (button == GLFW_MOUSE_BUTTON_LEFT) state->mouse_button_left = (action == GLFW_PRESS);
	if (button == GLFW_MOUSE_BUTTON_RIGHT) state->mouse_button_right = (action == GLFW_PRESS);
	if (button == GLFW_MOUSE_BUTTON_MIDDLE) state->mouse_button_middle = (action == GLFW_PRESS);
}

static void WindowSizeCallback(GLFWwindow* window, int width, int height)
{
	GLFW_State_t* state = (GLFW_State_t*)glfwGetWindowUserPointer(window);
	state->width = width;
	state->height = height;
	glViewport(0, 0, width, height);
}

GLFW_State_t * BackendSetup(const char* title, int width, int height)
{
	// Allocate state.
	GLFW_State_t* state = (GLFW_State_t*) malloc(sizeof(GLFW_State_t));

	memset(state, 0, sizeof(*state));

	// Set error callback.
	glfwSetErrorCallback(ErrorCallback);

	// Initialize.
	glfwInit();
	atexit(glfwTerminate);

	// Profile.
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	// Debug context.
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
	// Version.
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	// Double buffer.
	glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
	// Color buffer depth.
	glfwWindowHint(GLFW_RED_BITS, 8);
	glfwWindowHint(GLFW_GREEN_BITS, 8);
	glfwWindowHint(GLFW_BLUE_BITS, 8);
	glfwWindowHint(GLFW_ALPHA_BITS, 8);
	// Depth buffer depth.
	glfwWindowHint(GLFW_DEPTH_BITS, 16);
	// Anti-alias.
	glfwWindowHint(GLFW_SAMPLES, 4);

	// Create window.

	state->window = glfwCreateWindow(
			width,
			height,
			title,
			NULL,
			NULL
			);
	state->width = width;
	state->height = height;

	// Modify window.
#if GLFW_VERSION_MAJOR >= 3 && GLFW_VERSION_MINOR >= 3
	glfwSetWindowAttrib(state->window, GLFW_DECORATED, GLFW_TRUE);
	glfwSetWindowAttrib(
			state->window,
			GLFW_TRANSPARENT_FRAMEBUFFER,
			GLFW_TRUE
			);
#endif

	// Set window state pointer.
	glfwSetWindowUserPointer(state->window, state);

	// Make OpenGL context current.
	glfwMakeContextCurrent(state->window);

	// Enable vsync.
	glfwSwapInterval(1);

	// Set callbacks.
	glfwSetKeyCallback(state->window, KeyCallback);
	glfwSetCursorPosCallback(state->window, CursorPosCallback);
	glfwSetMouseButtonCallback(state->window, MouseButtonCallback);
	glfwSetWindowSizeCallback(state->window, WindowSizeCallback);

	// Return.
	return state;
}


void BackendFullscreen(GLFW_State_t *state, int enable)
{
	int width, height;
	glfwGetWindowSize(state->window, &width, &height);
	glfwSetWindowMonitor(
			enable ? state->window : 0,
			glfwGetPrimaryMonitor(),
			0, 0, width, height, 0
			);
}


void BackendDestroy(GLFW_State_t *state)
{
	glfwDestroyWindow(state->window);
	free(state);
}


void BackendInfo()
{
	printf("GLFW:\n");

	printf(
			"  Compile time version: %i.%i.%i (%s)\n",
			GLFW_VERSION_MAJOR,
			GLFW_VERSION_MINOR,
			GLFW_VERSION_REVISION,
			glfwGetVersionString()
			);

	{
		int major, minor, revision;
		glfwGetVersion(&major, &minor, &revision);
		printf(
				"  Run time version:     %i.%i.%i\n",
				major,
				minor,
				revision
				);
	}

	printf("  Available modes:\n");
	int monitor_count;
	GLFWmonitor** monitors = glfwGetMonitors(&monitor_count);
	for (int monitor_index = 0; monitor_index < monitor_count; ++monitor_index)
	{
		GLFWmonitor* monitor = monitors[monitor_index];
		printf("    %s:\n", glfwGetMonitorName(monitor));

		const GLFWvidmode* mode_current = glfwGetVideoMode(monitor);

		int mode_count;
		const GLFWvidmode* modes = glfwGetVideoModes(monitor, &mode_count);
		for (int mode_index = 0; mode_index < mode_count; ++mode_index)
		{
			const GLFWvidmode* mode = &modes[mode_index];
			int is_current = (
					mode->width == mode_current->width &&
					mode->height == mode_current->height &&
					mode->redBits == mode_current->redBits &&
					mode->greenBits == mode_current->greenBits &&
					mode->blueBits == mode_current->blueBits &&
					mode->refreshRate == mode_current->refreshRate
					);
			printf(
					"      %3d %4d %4d %3d RGB%d%d%d %s\n",
					mode_index + 1,
					mode->width,
					mode->height,
					mode->refreshRate,
					mode->redBits,
					mode->blueBits,
					mode->greenBits,
					is_current ? "*" : " "
					);
		}
	}
}


void BackendRun(
		GLFW_State_t *state,
		int (*update)(GLFW_State_t *state, float dt),
		void (*key)(GLFW_State_t *state, int key)
		)
{


	// Cast state.
	// Set input callbacks.
	state->key = key;
	// Handle events.
	while (glfwPollEvents(), !glfwWindowShouldClose(state->window))
	{
		// Handle time.
		static float t = 0.0;
		float t_prev = t;
		t = (float)glfwGetTime();
		float dt = (t_prev == 0.0) ? 0.0 : t - t_prev;


		// Update.
		if (update != nullptr && update(state, dt) == 1){
			glfwSwapBuffers(state->window);
		}
		else if(update != nullptr && update(state, dt) == -1){
			break;
		}
		else{
			t = 0.0;
		}

	}

}

void BackendWindowSize(GLFW_State_t *state, int* width, int*height)
{
	glfwGetWindowSize(state->window, width, height);
}

int BackendKey(GLFW_State_t *state, int key)
{
	return glfwGetKey(state->window, key) == GLFW_PRESS;
}

void BackendMouse(
		GLFW_State_t *state,
		int* button_left,
		int* button_right,
		int* button_middle,
		float* pos_x,
		float* pos_y,
		float* pos_prev_x,
		float* pos_prev_y
		)
{
	if (button_left) *button_left = state->mouse_button_left;
	if (button_right) *button_right = state->mouse_button_right;
	if (button_middle) *button_middle = state->mouse_button_middle;
	if (pos_x) *pos_x = state->mouse_pos_x;
	if (pos_y) *pos_y = state->mouse_pos_y;
	if (pos_prev_x) *pos_prev_x = state->mouse_pos_prev_x;
	if (pos_prev_y) *pos_prev_y = state->mouse_pos_prev_y;
}
