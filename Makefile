################################################################################
#Makefile to Generate Fractal GPU 	Edg@r J 2021 :)
################################################################################
#Compilers
GCC				= gcc
CXX 			= g++ 

CUDA 			= /usr/local/cuda-11.2
CUDA_SDK	= $(CUDA)/samples
NVCC     	= $(CUDA)/bin/nvcc

#Include Paths
CUDAINC   = -I. -I$(CUDA)/include -I$(CUDA_SDK)/common/inc 
INCQT			= -I/usr/include/x86_64-linux-gnu/qt5/

INC = -I$(HOME)/graphics/NumCpp/NumCpp/include
#Library Paths
CUDALIB		= -L/usr/lib/x86_64-linux-gnu -L$(CUDA)/lib64 \
						-lcuda -lcudart -lcudadevrt
GLLIB  		= -lGL -lGLU -lGLEW -lglfw
QTLIB			= -lQt5Core
OTHERLIB	= -lfreeimage -lstdc++fs 
LIB 			= $(CUDALIB) $(GLLIB) -lm -lstb

################ Choosing architecture code for GPU ############################
NVCC_ARCH			=
HOSTNAME		 	= $(shell uname -n)

ifeq ("$(HOSTNAME)","Edgar-PC")
	NVCC_ARCH		= -gencode arch=compute_61,code=sm_61
endif

###############	Debug, 0 -> False,  1-> True
DEBUGON						:= 1

ifeq (1,$(DEBUGON))
	CXXDEBUG 				:= -g -Wall
	CXXOPT					:= -O0 -std=c++17
#	NVCCDEBUG				:= -g -pg -G -fPIC -std=c++17
	NVCCDEBUG				:= 
	NVCCOPT					:= -O0
	NVCCFLAGSXCOMP 	:= -Xcompiler -g,-Wall,-O0 	
else
	CXXDEBUG 				:= 
	CXXOPT					:= -O3 -ffast-math -funroll-loops -std=c++17
	NVCCDEBUG				:= 
	NVCCOPT					:= -O3 --cudart=shared -use_fast_math
	NVCCFLAGSXCOMP 	:= -Xcompiler -O3,-ffast-math,-funroll-loops
endif
###############################################################################
CXXFLAGS				= $(CXXDEBUG) $(CXXOPT) -fopenmp -Wno-unused-function
NVCCFLAGS 			= $(NVCCDEBUG) $(NVCC_DP) --compile $(NVCCOPT) $(NVCC_ARCH)
NVCCFLAGSLINK		= $(NVCCDEBUG) $(NVCC_DP) $(NVCCOPT) $(NVCC_ARCH)
###############################################################################

TARGET = fracGPU

all: $(TARGET)

OBJLIST = shader.o text2D.o texture.o Accel.o WindowGL.o Render.o Fractal.o

fracGPU : main.o $(OBJLIST)
	$(NVCC) $(NVCCFLAGSLINK) $(NVCCFLAGSXCOMP) $(CUDAINC) $< -o $@ $(OBJLIST) $(LIB) 

main.o: Main.cpp Defs.hpp
	$(CXX) $(CXXFLAGS) $(INC) $(CUDAINC) -c $< -o $@ 



Accel.o: Accel.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCFLAGSXCOMP) $(INC) $(CUDAINC) $< -o $@ 

WindowGL.o: WindowGL.cpp WindowGL.hpp 
	$(CXX) $(CXXFLAGS) $(INC) $(CUDAINC) -c $< -o $@	

Render.o: Render.cpp Render.hpp
	$(CXX) $(CXXFLAGS) $(INC) $(CUDAINC) -c $< -o $@

Fractal.o: Fractal.cpp Fractal.hpp
	$(CXX) $(CXXFLAGS) $(INC) $(CUDAINC) -c $< -o $@



text2D.o: text2D.cpp text2D.hpp  
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@
	
shader.o: shader.cpp shader.hpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@	
	
texture.o: texture.cpp texture.hpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@	

clean:
	-rm -f *.o 
	-rm -f $(TARGET)

cleanDB:
	-rm -rf data/FractalDB*
