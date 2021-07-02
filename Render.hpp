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

#include "graphics.hpp"

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

// PNG 
#include <png.h>

//#include "spng.h"


const std::string cFrameBufferType[] = {"MAIN","FBO"};

class Render{

	private:
		Accel					*m_pGpu;
		Fractal				*m_pFrac;

		GLuint 				m_uiSimpleProgID;
		GLuint				m_uNumMappings;
		GLuint				m_uProjection;
		GLuint				m_uRotation;
		GLuint				m_uRenderType;
		GLuint				m_uRandom;
		
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
		unsigned int	m_uiFboWidth;
		unsigned int	m_uiFboHeight;

		glm::mat4			m_m4SaveRotation;
		int						m_iFps;
		int						m_rotationType;
		unsigned int	*m_uiSphereDitail;
		bool					m_bGenFrac;
		int						m_renderType;  
		float					m_randoms[2];   
		GLubyte				*m_glFrameBuffer;

		string				m_outPathcsv;
		string				m_outPathimg;
		string				m_rootDir;

		void 		renderToNormal(Settings cnfg);
		void 		renderToFBO(Settings cnfg);
		void 		drawALL(Settings cnfg);
		void 		drawInfo(Settings cnfg);
		void		setRootDirParams(string dir, int nclass, float density);
		void		setRootDirDataSet(string dir, int nclass, int count, Settings s);
		
		float   numPixel();
		void 		write_paramsto_csv(mapping *m, int numMaps,int count);
		void		savePNGfromOpenGLbuffer(int count);
		void		resizeGLbuffer(int w, int h);
		
		Render(unsigned int w, unsigned int h, Accel*& gpu, Fractal*& frac);
		~Render();
};

#if 0
static void screenshot_png(const char *filename, unsigned int width, unsigned int height,
        GLubyte **pixels, png_byte **png_bytes, png_byte ***png_rows) {
    size_t i, nvals;
    const size_t format_nchannels = 3;
    FILE *f = fopen(filename, "wb");
    nvals = format_nchannels * width * height;
    *pixels = realloc(*pixels, nvals * sizeof(GLubyte));
    *png_bytes = realloc(*png_bytes, nvals * sizeof(png_byte));
    *png_rows = realloc(*png_rows, height * sizeof(png_byte*));
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, *pixels);
    for (i = 0; i < nvals; i++)
        (*png_bytes)[i] = (*pixels)[i];
    for (i = 0; i < height; i++)
        (*png_rows)[height - i - 1] = &(*png_bytes)[i * width * format_nchannels];
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) abort();
    png_infop info = png_create_info_struct(png);
    if (!info) abort();
    if (setjmp(png_jmpbuf(png))) abort();
    png_init_io(png, f);
    png_set_IHDR(
        png,
        info,
        width,
        height,
        8,
        PNG_COLOR_TYPE_RGB,
    	PNG_INTERLACE_NONE,
    	PNG_COMPRESSION_TYPE_DEFAULT,
    	PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);
    png_write_image(png, *png_rows);
    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(f);
}
#endif

#endif /* RENDER_HPP */
