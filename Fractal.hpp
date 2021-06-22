/*
 * Fractal.hpp
 *
 *  Created on: May 31, 2021
 *      Author: Edg@r j.
 */
#ifndef FRACTAL_HPP
#define FRACTAL_HPP
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <random>

// Definitions
#include "Defs.hpp"
#include "Accel.cuh"

// FreeImage
#include <FreeImage.h>

// STB
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>


class Fractal {

  private:
    Accel*      m_pGPU;

    int         m_numMaps;
    int         m_numPoints;
    
  
  public:
    int         m_numIteration;
    int         m_numInstances;
    int         m_totalWeights;
    weights     *m_weights;
    int         m_numClass;
    int 				m_fileCount;
    mapping     *m_map;

    void    initFractalParam(Settings cfg, int count);
    void    appendWeights(int count);
    void    generateFractal();
    void    paramGenFromFile(int count);
    void    loadWeightsFromCSV();
    void    paramGenRandom();
    
    Fractal(Accel*& gpu, int numClass = DEFAULT_NUM_CLASS, int = DEFAULT_POINTS, int ffffffffff= DEFAULT_NUM_INST);
    inline int getNumOfMaps(){return m_numMaps;};
    inline int getNumOfPoints(){return m_numPoints;};
    inline int setNumOfPoints(int num){return m_numPoints = num;};

    ~Fractal();
    

};


#endif