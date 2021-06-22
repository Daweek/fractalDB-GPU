/*
 * WindowGL.cpp
 *
 *  Created on: May 31, 2021
 *      Author: Edg@r j.
 */
#include "WindowGL.hpp"

void WindowGL::generateRender(int w, int h){

	#ifdef GLEW_EGL
		// EGL initialization
	 // 1. Initialize EGL
  m_eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);

  EGLint major, minor;

  eglInitialize(m_eglDpy, &major, &minor);

  cout<<"EGL Major: "<<major<<"  Minor: "<<minor<<endl;
  // 2. Select an appropriate configuration
  EGLint numConfigs;
  EGLConfig eglCfg;

  //eglChooseConfig(m_eglDpy, configAttribs, &eglCfg, 1, &numConfigs);
  
  if( eglChooseConfig(m_eglDpy, configAttribs, &eglCfg, 1, &numConfigs)!= EGL_TRUE )
  {
      std:: cout << "ERROR: Configuration selection failed" << std::endl;
      exit(EXIT_FAILURE);
  }
  if( numConfigs == 0 )
  {
      std:: cout << "ERROR: No configurations" << std::endl;
      exit(EXIT_FAILURE);
  }

  // 3. Create a surface
  EGLSurface eglSurf = eglCreatePbufferSurface(m_eglDpy, eglCfg,pbufferAttribs);

  // 4. Bind the API
  eglBindAPI(EGL_OPENGL_API);

  // 5. Create a context and make it current
  EGLContext eglCtx = eglCreateContext(m_eglDpy, eglCfg, EGL_NO_CONTEXT,NULL);

  eglMakeCurrent(m_eglDpy, eglSurf, eglSurf, eglCtx);

  //glewExperimental = true;
	if (glewInit() != GLEW_OK)
		assert(!"GLEW initialization\n");
	if(!glewIsSupported("GL_EXT_framebuffer_object"))
		assert(!"The GL_EXT_framebuffer_object extension is required.\n");


  static const int MAX_DEVICES = 4;
  EGLDeviceEXT eglDevs[MAX_DEVICES];
  EGLint numDevices;

  PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
    (PFNEGLQUERYDEVICESEXTPROC)
    eglGetProcAddress("eglQueryDevicesEXT");

  printf("Detected %d devices\n", numDevices);

  eglQueryDevicesEXT(MAX_DEVICES, eglDevs, &numDevices);

	#else
			// glfw initialization
		glfwInit();
		glfwWindowHint(GLFW_SAMPLES,4);
		//glfwWindowHint(GLFW_VISIBLE,GLFW_FALSE);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
		
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
	#endif


}

WindowGL::WindowGL(unsigned int w, unsigned int h, Render*& rnd, Accel*& gpu,Fractal*& frac)  {

	// Select the enviroment to Draw
	generateRender(w,h);
	printf("............................OpenGL/EGL Related...........................\t\n");
	std::cout << glGetString( GL_VERSION ) << std::endl;
  std::cout << glGetString( GL_VENDOR ) << std::endl;
  std::cout << glGetString( GL_RENDERER ) << std::endl;
	printf("........................................................................\t\n\n");
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
	/*
	m_dTimeCurrent 	= glfwGetTime();
	m_iFpsCount++;

	if(m_dTimeCurrent - m_dTimeOld >= 1.0){
		m_pRender->m_iFps = m_iFpsCount;
		m_iFpsCount = 0;
		m_dTimeOld = m_dTimeCurrent;
	}
	*/
	// Draw the cube, the cross and the particles
	if(cnfg.fb == MAIN)	m_pRender->renderToNormal(cnfg);
	if(cnfg.fb == FBO)	m_pRender->renderToFBO(cnfg);

	// Swap the toilet...

	//for(int i=0;i<100;i++)
	#ifdef GLEW_EGL
		glFlush();
	#else
		glfwSwapBuffers(m_pWinID);
		glfwPollEvents();
	#endif
}

#ifndef GLEW_EGL
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
#endif

void WindowGL::resizeFrameBuffer(int w, int h){
	m_uiWinWidth = w;
	m_uiWinHeight = h;

	m_pRender->resizeGLbuffer(w,h);

	cout<<"Resize Frame Buffer"<<endl;

	#ifdef GLEW_EGL

	#else
			glfwSetWindowSize(m_pWinID, w, h);
	#endif
	


}


WindowGL::~WindowGL() {

	#ifdef GLEW_EGL
		std::cout<<"Closing EGL...\n";
		eglTerminate(m_eglDpy);
	#else
		std::cout<<"Closing glfw...\n";
		glfwWindowShouldClose(m_pWinID);
		glfwTerminate();
	#endif
}

