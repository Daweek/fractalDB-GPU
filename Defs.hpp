/*
 * Defs.hpp
 *
 *  Created on: May 31, 2021
 *      Author: Edg@r j.
 */
#ifndef DEFS_HPP
#define DEFS_HPP
#pragma once

// For OpenGL
#define GL_ON

// Memory related
#define MIN_THREADS 1
#define DEFAULT_THREADS 512
#define MAX_THREADS 1024

#define MIN_POINTS 1
#define DEFAULT_POINTS 100*1000
#define DEFAULT_NUM_CLASS 1000
#define MAX_POINTS 5*1000*1000 // 5 000 000

// Structs for points
typedef struct mp
{
	float a, b, c, d; // scaling/rotation matrix
	float x, y; // translation vertex
	float p; // mapping probability
} mapping;

typedef struct weigs
{
	float wa;
	float wb;
	float	wc;
	float	wd;
	float	we;
	float	wf;
} weights;

enum RenderType{POINTS_GRAY,POINTS_COLOR,PATCH_GRAY,PATCH_COLOR};
enum ParametersGen{FROM_CSV,FROM_RAND};
enum FrameBufferType{MAIN,FBO};
enum KernelType{NORMAL,DYNAMIC};


struct Settings{
		RenderType					rpt;
		FrameBufferType			fbt;
		KernelType					krt;
		ParametersGen				pgt;
};

using namespace std;

#endif // Header
