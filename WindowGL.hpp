/*
 * WindowGL.h
 *
 *  Created on: May 31, 2021
 *      Author: Edg@r j.
 */
#ifndef WINDOWGL_HPP
#define WINDOWGL_HPP
#pragma once

#include <iostream>
#include <assert.h>

#include "graphics.hpp"

#include "Defs.hpp"
#include "Render.hpp"
#include "Accel.cuh"
#include "Fractal.hpp"

#ifdef GLEW_EGL


#else


#endif
class WindowGL {

	private:
		Render*						m_pRender;
		Accel* 						m_pAccel;
		Fractal*					m_pFrac;

		bool							m_bContinue;
		bool							m_bPauseSim;

		#ifndef GLEW_EGL	
		inline static auto keyboardCallback(
			GLFWwindow *win,
			int key,
			int scancode,
			int action,
			int mods) -> void {
			WindowGL *window = static_cast<WindowGL*>(glfwGetWindowUserPointer(win));
			window->keyboard(key, scancode, action, mods);
		}
		#endif

		
	public:
		unsigned int			m_uiWinWidth;
		unsigned int			m_uiWinHeight;
		#ifdef GLEW_EGL
			EGLDisplay 				m_eglDpy;
			
		#else
			GLFWwindow* 			m_pWinID;
		#endif
		double						m_dTimeOld;
		double						m_dTimeCurrent;
		int								m_iFpsCount;
		

		WindowGL(unsigned int w, unsigned int h,Render*& rnd,Accel*& gpu, Fractal*& frac);
		void generateRender(int w, int h);
		void renderScene(Settings cnfg);

		#ifndef GLEW_EGL	
		auto keyboard(int key, int scancode, int action, int mods) -> void;
		#endif

		void resizeFrameBuffer(int w, int h);
		inline bool continueRender(){return m_bContinue;};
		inline bool pauseSimulation(){return m_bPauseSim;};
		~WindowGL();
};

#ifdef GLEW_EGL
static const EGLint configAttribs[] = {
          EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
          EGL_BLUE_SIZE, 8,
          EGL_GREEN_SIZE, 8,
          EGL_RED_SIZE, 8,
          EGL_DEPTH_SIZE, 8,
          EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
          EGL_NONE
};    

  static const int pbufferWidth = 512;
  static const int pbufferHeight = 512;

  static const EGLint pbufferAttribs[] = {
        
        EGL_WIDTH, pbufferWidth,
        EGL_HEIGHT, pbufferHeight,
        EGL_NONE,
  };
#endif

#endif /* WINDOWGL_HPP */
