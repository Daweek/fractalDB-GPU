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


class Accel {
	private:
		CUdevice						m_cuDevice;
		cudaDeviceProp			m_cuDevProp;

		int 		m_driverVersion;
		int			m_runtimeVersion;
		int			m_numBlocks;
		int			m_blockSize;
		
		
		float  		m_kernel_mili; 
		float			*d_glmap;
		float			*d_glPoss;
		float			*d_glColor;
		float4		*d_borders;
		float			*h_borders;
		int				*d_mutex;

		mapping	*d_map;
	
		struct cudaGraphicsResource* g_strucMapVBO;
		struct cudaGraphicsResource* g_strucPoss;
		struct cudaGraphicsResource* g_strucColor;


	public:
		float		m_fFlops;
		float		m_fStepsec;

		GLuint 	g_mapVBO;
		GLuint	g_poss;
		GLuint	g_color;

		float		m_fXmin;
		float		m_fXmax;
		float		m_fYmin;
		float		m_fYmax;

		bool		m_bChangeMalloc;
		bool		m_bChangeInterop;

		Accel();
		void fractalKernel(int numMappings, int numPoints);
		void malloCUDA(mapping *mapped, int numMaps);
		void interopCUDA();
		~Accel();


};


/*
	// Originl Kernell
	__global__ void kernel(float4* d_pointData, int numPoints,mapping *d_mappings, int numMappings)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		// If needed for performance, move curand_init to seperate kernel and store
		// states in device memory
		curandState state;
		curand_init((unsigned long long) clock(), index, 0, &state);

		// Set up transformation mapping once per block in shared memory
		extern __shared__ mapping maps[];
		if(threadIdx.x == 0)
		{
			#pragma unroll
			for(int i = 0; i < numMappings; i++)
					maps[i] = d_mappings[i];
		}
		__syncthreads();

		// Initially start at a mapping vertex to guarantee we stay inside the
		// iterated function system
		int currentTarget = index % numMappings;
		float2 currentPosition, newPosition;
		currentPosition.x = maps[currentTarget].x;
		currentPosition.y = maps[currentTarget].y;

		for(int i = index; i < numPoints; i += stride)
		{
			// set the current vertex to the currentPosition
			d_pointData[i].x = currentPosition.x;
			d_pointData[i].y = currentPosition.y;

			// set the iteration percentage and current target mapping
			d_pointData[i].z =  i / (float) numPoints;
			d_pointData[i].w = currentTarget;

			// find random target with given mapping probabilities
			// If needed for performance, find method to remove thread divergence
			// Note: changing 4 to numMappings in for loop reduced performance 50%
			float currentProb = curand_uniform(&state);
			float totalProb = 0.0f;
			for(int j = 0; j < numMappings; j++)
			{
					totalProb += maps[j].p;
					if(currentProb < totalProb)
					{
							currentTarget = j;
							break;
					}
			}

			// calculate the transformation
			// (x_n+1) = (a b)(x_n) + (e)
			// (y_n+1)   (c d)(y_n)   (f)
			newPosition.x = maps[currentTarget].a * currentPosition.x +
											maps[currentTarget].b * currentPosition.y +
											maps[currentTarget].x;
			newPosition.y = maps[currentTarget].c * currentPosition.x +
											maps[currentTarget].d * currentPosition.y +
											maps[currentTarget].y;
			currentPosition = newPosition;
		}

	}
*/

#endif /* ACCEL_CUH */
