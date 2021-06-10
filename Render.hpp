/*
 * Render.hpp
 *
 *  Created on: May 31, 2021
 *      Author: Edg@r j.
 */
#ifndef RENDER_HPP
#define RENDER_HPP
#pragma once

#include <iostream>
#include <assert.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <filesystem>

#include <GL/glew.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Defs.hpp"
#include "Accel.cuh"
#include "Fractal.hpp"
#include "shader.hpp"
#include "texture.hpp"
#include "text2D.hpp"

// STB
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>


const std::string cFrameBufferType[] = {"MAIN","FBO"};
const std::string cRenderParticleType[] = {"POINTS","SPHERE","TEXTURE"};

class Render{

	private:
		Accel					*m_pGpu;
		Fractal				*m_pFrac;

		unsigned int	m_uiFboWidth;
		unsigned int	m_uiFboHeight;

		GLuint 				m_uiSimpleProgID;
		GLuint				m_uNumMappings;
		GLuint				m_uProjection;
		GLuint				m_uRotation;
	
		GLuint				m_uiTextureProgID;
		GLuint				m_uiTextureID;
		GLuint				m_uiQuadVertexBuffer;

		static const
		GLfloat  			m_fQuadVertexBufferData[18];

		GLuint				m_uiFboTexture;
		GLuint				m_uiFboDepth;
		GLuint				m_uiFboFramBuff;

		GLuint				m_uiVertArrayID;
		GLenum				m_eDrawBuffers[1];

				
	public:
		glm::mat4			m_m4SaveRotation;
		int						m_iFps;
		int						m_rotationType;
		unsigned int	*m_uiSphereDitail;
		bool					m_bGenFrac;
		GLubyte				*m_glFrameBuffer;

		string				m_outPathcsv;
		string				m_outPathimg;
		string				m_rootDir;

		void 		renderToNormal(Settings cnfg);
		void 		renderToFBO(Settings cnfg);
		void 		drawALL(Settings cnfg);
		void 		drawInfo(Settings cnfg);
		void		setRootDirParams(string dir, int nclass, float density);
		void		setRootDirDataSet(string dir, int nclass, int count);
		
		float   numPixel();
		void 		write_paramsto_csv(mapping *m, int numMaps,int count);
		void		savePNGfromOpenGLbuffer(int count);
		
		Render(unsigned int w, unsigned int h, Accel*& gpu, Fractal*& frac);
		~Render();
};

#endif /* RENDER_HPP */
