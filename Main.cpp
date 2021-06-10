/*
 * Main.cpp
 *
 *  Created on: May 31, 2021
 *      Author: Edg@r j.
 */
#include "Defs.hpp"
#include "Accel.cuh"
#include "Render.hpp"
#include "WindowGL.hpp"
#include "Fractal.hpp"



// Main Objects
Accel*							g_oGPU;
WindowGL*						g_oWindow;
Render*							g_oRender;
Fractal*						g_oFrac;

// General options for Hardware & Render
Settings						g_sConfig;

// Window
unsigned int 				g_WinWidth  = 512;
unsigned int 				g_WinHeight = 512;

int main(int argc, char **argv)
{
	// TODO catch main arguments
	cout<<"Fractal DB Generator...\n";
	nc::random::seed(1);

	// Main Variables for object options
	g_sConfig.fbt	= MAIN;						// MAIN,FBO
	g_sConfig.rpt = POINTS_GRAY;  	// POINTS,SPHERE
	g_sConfig.krt = NORMAL;					// NORMAL,DYNAMIC --> for DYNAMIC remember -rdc=true for compiling
	g_sConfig.pgt = FROM_RAND;			// FROM_CSV or FROM_RAND


	// CUDA Init..
	g_oGPU		= new Accel();

	// Fractal object settings
	g_oFrac 	= new Fractal(g_oGPU);
	cout<<"\nNum of Points: "<<g_oFrac->getNumOfPoints()<<", Num of Classes: "<<g_oFrac->m_numClass<<endl;

	// Init GLFW and OpenGL
	g_oWindow = new	WindowGL(g_WinWidth,g_WinHeight,g_oRender,g_oGPU,g_oFrac);

		
	g_oGPU->interopCUDA();
	//cout<<"Before Fractal Init \n\n\n";
	
	//g_oFrac->initFractalParam(g_sConfig);


	// Allocate CUDA memory
	//g_oGPU->malloCUDA(g_oFrac->m_map, g_oFrac->getNumOfMaps());

	// Prepare paths for output
	g_oRender->setRootDirParams("data/",g_oFrac->m_numClass,0.2f);

	// Read the weights
	g_oFrac->loadWeightsFromCSV();

	// First Run ()
	//g_oFrac->generateFractal();
	//g_oWindow->renderScene(g_sConfig);
	//g_oRender->numPixel();
	// Prepare for FPS measure
	//g_oWindow->m_dTimeOld = glfwGetTime();

	// For Debuggin purposes
	#if 0  
		while (g_oWindow->continueRender() && !glfwWindowShouldClose(g_oWindow->m_pWinID)){

			if(g_oRender->m_bGenFrac){
				do{
					g_oFrac->initFractalParam(g_sConfig);
					g_oGPU->malloCUDA(g_oFrac->m_map, g_oFrac->getNumOfMaps());
					g_oFrac->generateFractal();
					g_oWindow->renderScene(g_sConfig);
				}while(g_oRender->numPixel() < 0.2);

				// Write parameters to CSV
				//stringstream ss;
				//ss<<setw(5)<<setfill('0')<<to_string(num);
				//string s = ss.str();
				//cout<<s<<endl;

				//string filecsv = outPathcsv +"/" + s + ".csv";
				//cout<<filecsv<<endl;
				//g_oRender->write_paramsto_csv(filecsv,g_oFrac->m_map,g_oFrac->getNumOfMaps());
							
				//string fileimg = outPathimg + "/" + s + ".png";
				//cout<<fileimg<<endl;
				//g_oRender->savePNGfromOpenGLbuffer(fileimg.c_str(),g_oWindow->m_pWinID);

				//cout<<"save: " << s << endl;
				//num++;

				g_oRender->m_bGenFrac	= false;
			}

			//if(g_oWindow->pauseSimulation() == false)
				//g_oFrac->generateFractal();
			glfwPollEvents();
			//g_oWindow->renderScene(g_sConfig);
			
		}

	#else
	int count = 0;
	while(count < g_oFrac->m_numClass){
		do{
					g_oFrac->initFractalParam(g_sConfig,count);
					g_oGPU->malloCUDA(g_oFrac->m_map, g_oFrac->getNumOfMaps());
					g_oFrac->generateFractal();
					g_oWindow->renderScene(g_sConfig);
					//g_oFrac->m_fileCount++;
		}while(g_oRender->numPixel() < 0.2);

		//cout<<filecsv<<endl;
		g_oRender->write_paramsto_csv(g_oFrac->m_map,g_oFrac->getNumOfMaps(),count);
					
		
		//cout<<fileimg<<endl;
		g_oRender->savePNGfromOpenGLbuffer(count);

		stringstream ss;
		ss<<setw(5)<<setfill('0')<<to_string(count);
		string s = ss.str();
		cout<<"Generic Params save: " << s << endl;
		count++;
	}
	#endif


	cout<<"\nContinue to generate the DataBase...\n\n"<<endl;
	g_sConfig.pgt = FROM_CSV;
	g_oFrac->m_numIteration = 200000;
	g_oFrac->setNumOfPoints(200000);
	g_oRender->setRootDirDataSet("data/",g_oFrac->m_numClass,-1);

	//int t = 0;
	for (int nclass = 0; nclass < g_oFrac->m_numClass; nclass++){
	//for (int nclass = 0; nclass < 1; nclass++){

		//t = nclass * g_oFrac->m_totalWeights;
		
		for (int nweights = 0; nweights < g_oFrac->m_totalWeights; nweights++){
		//for (int nweights = 0; nweights < 1; nweights++){
			g_oRender->setRootDirDataSet("data/",g_oFrac->m_numClass,nclass);
			g_oFrac->initFractalParam(g_sConfig,nclass);
			g_oFrac->appendWeights(nweights);
			g_oGPU->malloCUDA(g_oFrac->m_map, g_oFrac->getNumOfMaps());
			g_oFrac->generateFractal();
			//cout<<"Weights: "<<nweights<<endl;

			for(int rot = 0; rot < 4; rot++){
				g_oRender->m_rotationType = rot;
				g_oWindow->renderScene(g_sConfig);
				g_oRender->numPixel();
				g_oRender->savePNGfromOpenGLbuffer(nweights*4 + rot);
			}
			
		}
		stringstream ss;
		ss<<setw(5)<<setfill('0')<<to_string(nclass);
		string s = ss.str();
		cout<<"DB Class save: " << s << endl;	
	}


	cout<<"\nCleaning objects...\n";
	delete g_oGPU;
	delete g_oRender;
	delete g_oFrac;
	delete g_oWindow;

	return 0;
}
