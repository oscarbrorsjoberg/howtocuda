#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "gltex.hpp"

GLuint *create_texture_id(
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
