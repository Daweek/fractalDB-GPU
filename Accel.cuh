/*
 * Accel.h
 *
 *  Created on: May 31, 2021
 *      Author: Edg@r j.
 */
#ifndef ACCEL_CUH
#define ACCEL_CUH
#pragma once

#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "graphics.hpp"

// ***** CUDA includes
#include <cuda.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

#include "Defs.hpp"

typedef struct f
{
	// GPU pointers
	float			*d_glPoss;
	float			*d_glColor;
	float4		*d_borders;
	
	int				*d_mutex;
	mapping		*d_map;

	// CPU pointers
	GLuint		g_poss;
	GLuint		g_color;
	float			*h_borders;
	mapping		*h_map;

	struct cudaGraphicsResource* g_strucPoss;
	struct cudaGraphicsResource* g_strucColor;

	float		fXmin;
	float		fXmax;
	float		fYmin;
	float		fYmax;

}fraktal;



class Accel {
	private:
		CUdevice						m_cuDevice;
		cudaDeviceProp			m_cuDevProp;

		int 		m_driverVersion;
		int			m_runtimeVersion;
		int			m_numBlocks;
		int			m_blockSize;
		
		
		float  		m_kernel_mili; 
		
	public:
		fraktal m_fk[NF];
		float		m_fFlops;
		float		m_fStepsec;

		bool		m_bChangeMalloc;
		bool		m_bChangeInterop;

		Accel();
		void fractalKernel(int numMappings, int numPoints);
		void malloCUDA(int numMaps);
		void interopCUDA();
		~Accel();

};


#endif /* ACCEL_CUH */
