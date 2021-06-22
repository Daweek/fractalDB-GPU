/*
 * Graphics.hpp
 *
 *  Created on: June 22, 2021
 *      Author: Edg@r j.
 */
#ifndef GRAPHICS_HPP
#define GRAPHICS_HPP

#ifdef GLEW_EGL

  #include "GL/glew.h"
  #include <EGL/egl.h>
  #include <EGL/eglext.h>

#else

  #include <GL/glew.h>
  #include <GLFW/glfw3.h>

#endif

#endif // Header