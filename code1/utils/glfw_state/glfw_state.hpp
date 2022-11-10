#ifndef UTIL_BACKEND_H
#define UTIL_BACKEND_H

#include <GLFW/glfw3.h>

struct GLFW_State_t {
  GLFWwindow* window;
  int width;
  int height;
  void (*key)(GLFW_State_t *backend_state, int key);
  void (*mouse)(GLFW_State_t *backend_state, int key);
  int mouse_button_left;
  int mouse_button_right;
  int mouse_button_middle;
  float mouse_pos_x;
  float mouse_pos_y;
  float mouse_pos_prev_x;
  float mouse_pos_prev_y;
};

GLFW_State_t* BackendSetup(
    const char* title,
    int width,
    int height
);

void BackendDestroy(GLFW_State_t *backend_state);

void BackendFullscreen(GLFW_State_t *backend_state, int enable);

void BackendInfo();

void BackendRun(
    GLFW_State_t *backend_state,
    int (*update)(GLFW_State_t *backend_state, float dt),
    void (*key)(GLFW_State_t *backend_state, int key)
);

void BackendWindowSize(GLFW_State_t *backend_state, int* width, int*height);

int BackendKey(GLFW_State_t *backend_state, int key);

void BackendMouse(
    GLFW_State_t * backend_state,
    int* button_left,
    int* button_right,
    int* button_middle,
    float* pos_x,
    float* pos_y,
    float* pos_prev_x,
    float* pos_prev_y
);




#endif 
