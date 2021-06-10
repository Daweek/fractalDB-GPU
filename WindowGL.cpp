/*
 * WindowGL.cpp
 *
 *  Created on: May 31, 2021
 *      Author: Edg@r j.
 */
#include "WindowGL.hpp"

WindowGL::WindowGL(unsigned int w, unsigned int h, Render*& rnd, Accel*& gpu,Fractal*& frac)  {

	// glfw initialization
	glfwInit();
	glfwWindowHint(GLFW_SAMPLES,4);
	//glfwWindowHint(GLFW_VISIBLE,GLFW_FALSE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	
	m_pWinID = glfwCreateWindow(w, h, "FractalCreator Visor",NULL,NULL);
	glfwMakeContextCurrent(m_pWinID);

	// Glew Initialization
	glewExperimental = true;
	if (glewInit() != GLEW_OK)
		assert(!"GLEW initialization\n");
	if(!glewIsSupported("GL_EXT_framebuffer_object"))
		assert(!"The GL_EXT_framebuffer_object extension is required.\n");

	// Callbacks for glfw and hints for window
	//glfwSetWindowPos(m_pWinID,12,12);
	glfwSetWindowPos(m_pWinID,5400,2100);
	glfwSetWindowUserPointer(m_pWinID, this);
	glfwSetKeyCallback(m_pWinID,keyboardCallback);
	//glfwSetMouseButtonCallback(m_pWinID,mouseCallback);
	//glfwSetCursorPosCallback(m_pWinID,motionCallback);

	//glfwSetInputMode(m_pWinID,GLFW_STICKY_KEYS,GLFW_STICKY_KEYS);
	glfwSetInputMode(m_pWinID,GLFW_CURSOR,GLFW_CURSOR_NORMAL);

	// Construct the render object
	rnd = new Render(w,h,gpu,frac);

	// Assign values to object memory
	m_bContinue		= true;
	m_bPauseSim		= false;
	m_uiWinWidth  = w;
	m_uiWinHeight =	h;
	

	//Values for measuring FPS
	m_dTimeCurrent 	= m_dTimeOld = 0.0;
	m_iFpsCount			= 0;

	// Get a handler for ...
	m_pRender = rnd;
	m_pAccel	= gpu;
	m_pFrac		= frac;	
	
}

void WindowGL::renderScene(Settings cnfg){

	// Measure FPS performance
	m_dTimeCurrent 	= glfwGetTime();
	m_iFpsCount++;

	if(m_dTimeCurrent - m_dTimeOld >= 1.0){
		m_pRender->m_iFps = m_iFpsCount;
		m_iFpsCount = 0;
		m_dTimeOld = m_dTimeCurrent;
	}
	
	// Draw the cube, the cross and the particles
	if(cnfg.fbt == MAIN)	m_pRender->renderToNormal(cnfg);
	if(cnfg.fbt == FBO)		m_pRender->renderToFBO(cnfg);

	// Swap the toilet...

	//for(int i=0;i<100;i++)	
	glfwSwapBuffers(m_pWinID);
	glfwPollEvents();
}


auto WindowGL::keyboard(int key, int scancode, int action, int mods) -> void {
	// For only press one time....even we hold
	if(action == GLFW_PRESS){
		// Show Keyboard capabilities...
		if(key == GLFW_KEY_SLASH){
			if(mods == GLFW_MOD_SHIFT){
				std::cout<<"\nKeyboard Capabilities:"<<std::endl
								 <<" q,ESC\t  --> Exit"<<std::endl;
	 		}
		}
		// Quit
		if(key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE){
			m_bContinue = false;
		}
		// Pause simulation...
		if(key == GLFW_KEY_Z){
			m_bPauseSim = !m_bPauseSim;
		}

		if(key == GLFW_KEY_D){
			m_pRender->m_bGenFrac = true;
			m_pFrac->m_fileCount++;
		}
		if(key == GLFW_KEY_A){
			if(m_pFrac->m_fileCount>0){
				m_pFrac->m_fileCount--;
				m_pRender->m_bGenFrac = true;
			}
		}

	}
}

WindowGL::~WindowGL() {

	std::cout<<"Closing glfw...\n";
	glfwWindowShouldClose(m_pWinID);
	glfwTerminate();

}

