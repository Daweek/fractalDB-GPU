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
#include "cxxopts.hpp"


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

unsigned int				g_renderW		= 256;
unsigned int				g_renderH		= 256;

// Strings
const string cRenderType[] = {"GRAY","COLOR"};
const string cRenderFilter[] = {"POINTS","PATCH"};


int main(int argc, char **argv)
{
	// TODO catch main arguments
	cxxopts::Options options("Fractal Generator", "Fractal Database generator using CUDA-OpenGL");

	options.add_options()
		("d,debug", "Enable debugging") // a bool parameter
		("c,class", "Number of classes", cxxopts::value<int>()->default_value("1"))
		("i,instances", "Number of instances per Class... E.g. 1 create 100, 10 create 1,000", cxxopts::value<int>()->default_value("1"))
		("n,npoints","Number of points for to search fractal", cxxopts::value<int>()->default_value("100000"))
		("r,color", "Render type, if not set will render to gray", cxxopts::value<bool>()->default_value("false"))
		("p,patch", "Render filter, if not set render with pixel-points", cxxopts::value<bool>()->default_value("false"))
		;

	auto args = options.parse(argc, argv);

	// Configure according to arguments
	if (args["color"].as<bool>()) g_sConfig.rt = COLOR;
	else													g_sConfig.rt = GRAY;

	if (args["patch"].as<bool>()) g_sConfig.rf = PATCH;
	else													g_sConfig.rf = POINTS;

	// Main Variables for object options

	g_sConfig.fb	= MAIN;						// MAIN,FBO
	g_sConfig.pa	= FROM_RAND;			// FROM_CSV or FROM_RAND

	// Echo parameters
	cout<<"Fractal DB Generator...\n"<<endl;
	cout<<"Number of classes:"<<args["class"].as<int>()<<endl;
	cout<<"Number of instances per Class:"<<args["instances"].as<int>()<<endl;
	cout<<"Number of points:"<<args["npoints"].as<int>()<<endl;
	cout<<"Render type:"<<cRenderType[g_sConfig.rt]<<endl;
	cout<<"Render filter type:"<<cRenderFilter[g_sConfig.rf]<<endl<<endl;


	// CUDA Init..
	g_oGPU		= new Accel();

	// Fractal object settings
	g_oFrac 	= new Fractal(g_oGPU, args["class"].as<int>(), args["npoints"].as<int>(),args["instances"].as<int>());

	// Init GLFW and OpenGL
	g_oWindow = new	WindowGL(g_WinWidth,g_WinHeight,g_oRender,g_oGPU,g_oFrac);
		
	g_oGPU->interopCUDA();
	
	// Prepare paths for output
	g_oRender->setRootDirParams("data/",g_oFrac->m_numClass,0.2f);

	// For Debuggin purposes
	#if 1
		int count = 0;  
		while (g_oWindow->continueRender() && !glfwWindowShouldClose(g_oWindow->m_pWinID)){

			if(g_oRender->m_bGenFrac){
				//do{
					g_oFrac->initFractalParam(g_sConfig,count);
					g_oGPU->malloCUDA(g_oFrac->getNumOfMaps());
					g_oFrac->generateFractal();
					g_oWindow->renderScene(g_sConfig);
					glfwPollEvents();
				//}while(g_oRender->numPixel() < 0.02);

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
	// Always search for the params in Point-Gray
	g_oRender->m_renderType=0;
	while(count < g_oFrac->m_numClass){
		do{
					g_oFrac->initFractalParam(g_sConfig,count);
					g_oGPU->malloCUDA(g_oFrac->getNumOfMaps());
					g_oFrac->generateFractal();
					g_oWindow->renderScene(g_sConfig);
					//g_oFrac->m_fileCount++;
		}while(g_oRender->numPixel() < 0.2);

		//cout<<filecsv<<endl;
		g_oRender->write_paramsto_csv(g_oFrac->getNumOfMaps(),count);
					
		//cout<<fileimg<<endl;
		g_oRender->savePNGfromOpenGLbuffer(count);

		stringstream ss;
		ss<<setw(5)<<setfill('0')<<to_string(count);
		string s = ss.str();
		cout<<"Generic Params save: " << s << endl;
		count++;
	}
	
	//sleep(2);
	cout<<"\nContinue to generate the DataBase...\n\n"<<endl;

	// Re-shape the framebuffer
	//g_oWindow->resizeFrameBuffer(128,128);
	g_oWindow->resizeFrameBuffer(g_renderW,g_renderH);
	//g_oWindow->resizeFrameBuffer(362,362);
	//sleep(2);

	// Configure shader render type
	if			(g_sConfig.rf == POINTS && g_sConfig.rt == GRAY)g_oRender->m_renderType =0;
	else if	(g_sConfig.rf == POINTS && g_sConfig.rt == COLOR)g_oRender->m_renderType=2;
	else if	(g_sConfig.rf == PATCH && g_sConfig.rt == GRAY)g_oRender->m_renderType=1;
	else if	(g_sConfig.rf == PATCH && g_sConfig.rt == COLOR)g_oRender->m_renderType=3;

	cout<<"Render type:"<<g_oRender->m_renderType<<endl;

	
	// Read the weights
	g_oFrac->loadWeightsFromCSV();
	
	g_sConfig.pa = FROM_CSV;
	//g_oFrac->m_numIteration = 100000;
	g_oFrac->setNumOfPoints(250000);
	g_oRender->setRootDirDataSet("data/",g_oFrac->m_numClass,-1,g_sConfig);

	//int t = 0;
	for (int nclass = 0; nclass < g_oFrac->m_numClass; nclass++){
	//for (int nclass = 0; nclass < 1; nclass++){
		//t = nclass * g_oFrac->m_totalWeights;
		for(int ins=0;ins<g_oFrac->m_numInstances;ins++){
			// Create random patch
			g_oRender->m_randoms[0] = (float) rand()/RAND_MAX;
			g_oRender->m_randoms[1] = (float) rand()/RAND_MAX;

			for (int nweights = 0; nweights < g_oFrac->m_totalWeights; nweights++){
			//for (int nweights = 0; nweights < 1; nweights++){
				g_oRender->setRootDirDataSet("data/",g_oFrac->m_numClass,nclass,g_sConfig);
				g_oFrac->initFractalParam(g_sConfig,nclass);
				g_oFrac->appendWeights(nweights);
				g_oGPU->malloCUDA(g_oFrac->getNumOfMaps());
				g_oFrac->generateFractal();
				//cout<<"Weights: "<<nweights<<endl;
				//sleep(2000); 
				for(int rot = 0; rot < 4; rot++){
					g_oRender->m_rotationType = rot;
					g_oWindow->renderScene(g_sConfig);
					g_oRender->numPixel();
					g_oRender->savePNGfromOpenGLbuffer(nweights * 4 + rot + ins*100);
					//sleep(1); 
					
				}	
			}
		}

		stringstream ss;
		ss<<setw(5)<<setfill('0')<<to_string(nclass);
		string s = ss.str();
		cout<<"DB Class save: " << s << endl;	

	#if 0
	// Test
	const int width = 362;
 	const int height = 362;
	#define CHANNEL_NUM 3

	/*** NOTICE!! You have to use uint8_t array to pass in stb function  ***/
	// Because the size of color is normally 255, 8bit.
	// If you don't use this one, you will get a weird imge.
	uint8_t* pixels = new uint8_t[width * height * CHANNEL_NUM];

	int index = 0;
	for (int j = height - 1; j >= 0; --j)
	{
		for (int i = 0; i < width; ++i)
		{
		float r = (float)i / (float)width;
		float g = (float)j / (float)height;
		float b = 0.2f;
		int ir = int(255.99 * r);
		int ig = int(255.99 * g);
		int ib = int(255.99 * b);

		pixels[index++] = ir;
		pixels[index++] = ig;
		pixels[index++] = ib;
		}
	}
	
	// if CHANNEL_NUM is 4, you can use alpha channel in png
	stbi_write_png("stbpng.png", width, height, CHANNEL_NUM, pixels, width * CHANNEL_NUM);

	// You have to use 3 comp for complete jpg file. If not, the image will be grayscale or nothing.
	stbi_write_jpg("stbjpg3.jpg", width, height, 3, pixels, 100);
	
	
	delete[] pixels;
	#endif
	

	}
#endif

	cout<<"\nCleaning objects...\n";
	delete g_oGPU;
	delete g_oRender;
	delete g_oFrac;
	delete g_oWindow;

	return 0;
}
