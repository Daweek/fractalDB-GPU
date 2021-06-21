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

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Defs.hpp"
#include "Render.hpp"
#include "Accel.cuh"
#include "Fractal.hpp"


class WindowGL {

	private:
		Render*						m_pRender;
		Accel* 						m_pAccel;
		Fractal*					m_pFrac;

		bool							m_bContinue;
		bool							m_bPauseSim;

			
		inline static auto keyboardCallback(
			GLFWwindow *win,
			int key,
			int scancode,
			int action,
			int mods) -> void {
			WindowGL *window = static_cast<WindowGL*>(glfwGetWindowUserPointer(win));
			window->keyboard(key, scancode, action, mods);
		}

		
	public:
		unsigned int			m_uiWinWidth;
		unsigned int			m_uiWinHeight;
		GLFWwindow* 			m_pWinID;
		double						m_dTimeOld;
		double						m_dTimeCurrent;
		int								m_iFpsCount;
		

		WindowGL(unsigned int w, unsigned int h,Render*& rnd,Accel*& gpu, Fractal*& frac);
		void renderScene(Settings cnfg);
		auto keyboard(int key, int scancode, int action, int mods) -> void;
		void resizeFrameBuffer(int w, int h);
		inline bool continueRender(){return m_bContinue;};
		inline bool pauseSimulation(){return m_bPauseSim;};
		~WindowGL();
};

#endif /* WINDOWGL_HPP */
