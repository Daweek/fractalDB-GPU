/*
 * Render.cpp
 *
 *  Created on: May 31, 2021
 *      Author: Edg@r j.
 */
#include "Render.hpp"
// Prepare static coordinates for camera
const GLfloat Render::m_fQuadVertexBufferData[18] = {
				-1.0f, -1.0f, 0.0f,
				 1.0f, -1.0f, 0.0f,
				-1.0f,  1.0f, 0.0f,
				-1.0f,  1.0f, 0.0f,
				 1.0f, -1.0f, 0.0f,
				 1.0f,  1.0f, 0.0f,
};
// Strings
const string cRenderType[] = {"GRAY","COLOR"};
const string cRenderFilter[] = {"POINTS","PATCH"};

Render::Render(	unsigned int w, unsigned int h, Accel*& gpu, Fractal*& frac) {

	std::cout<<"Loading shaders and preparing OpenGL buffers...\n";

	// Create Color Attachment for frame buffer
	glGenTextures(1, &m_uiFboTexture);
	glBindTexture(GL_TEXTURE_RECTANGLE, m_uiFboTexture);
	glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	// Create Depth Attachment for frame buffer
	glGenRenderbuffers( 1, &m_uiFboDepth );
	glBindRenderbuffer( GL_RENDERBUFFER, m_uiFboDepth );
	glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT, w, h);
	glBindRenderbuffer( GL_RENDERBUFFER, 0 );

	// Create Frame buffer
	glGenFramebuffers(1, &m_uiFboFramBuff);
	glBindFramebuffer(GL_FRAMEBUFFER, m_uiFboFramBuff);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_uiFboTexture, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_uiFboDepth);

	// Check Frame buffer status
	if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		assert(!"Framebuffer is incomplete.\n");
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Load shaders for text
	initText2D("shaders/Font.DDS");

	// Load simple shaders for Fractals
	m_uiSimpleProgID 	= LoadShaders("shaders/fractalvert.glsl","shaders/fractalfrag.glsl");
	m_uNumMappings		= glGetUniformLocation(m_uiSimpleProgID, "numMappings");
	m_uProjection			= glGetUniformLocation(m_uiSimpleProgID, "projection");
	m_uRotation				= glGetUniformLocation(m_uiSimpleProgID, "rotation");
	m_uRenderType			= glGetUniformLocation(m_uiSimpleProgID, "rtype");
	m_uRandom					= glGetUniformLocation(m_uiSimpleProgID, "rnd");
	

	// Prepare shaders for FBO presentation on Texture
	glGenBuffers(1, &m_uiQuadVertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_uiQuadVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_fQuadVertexBufferData), m_fQuadVertexBufferData, GL_STATIC_DRAW);

	m_uiTextureProgID = LoadShaders( "shaders/TextureFBOVert.glsl", "shaders/TextureFBOFrag.glsl" );
	m_uiTextureID 		= glGetUniformLocation(m_uiTextureProgID, "renderedTexture");

	// OpenGL options and hints
	glEnable(GL_POINTS);
	glDisable(GL_POINT_SMOOTH);
	glDisable(GL_MULTISAMPLE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	glEnable(GL_MINMAX);
	glMinmax(GL_MINMAX, GL_RGB, GL_FALSE);


	glGenVertexArrays(1,&m_uiVertArrayID);
	glBindVertexArray(m_uiVertArrayID);

	m_eDrawBuffers[0] = {GL_COLOR_ATTACHMENT0};
	glDrawBuffers(1, m_eDrawBuffers); // "1" is the size of DrawBuffers
	
	// Pass information for the main object
	m_uiFboWidth 	= w;
	m_uiFboHeight	= h;

	// Prepare the framebuffer
	m_glFrameBuffer = (GLubyte*) malloc ( 3 * sizeof(GLubyte) * w * h);
	
	// Holder for GPU object
	m_pGpu 	= gpu;
	m_pFrac	= frac;

	// Timer
	m_iFps = 0;
	m_bGenFrac = true;

	m_rotationType = 0;
	m_renderType = 0;
}


void Render::drawInfo(Settings cnfg){
	// Print some Text
	std::string buf;
	std::stringstream st1,st2,st3;
	int x,y,size;

	size	= 28;
	x 		= 10;
	y			= 570;

	
	y		-= size*12;
	buf = "FPS: "+ std::to_string(m_iFps);
	printText2D(buf.c_str(),x,y,size);

	y		-= size;
	buf = "Sec/Step: " + std::to_string(m_pGpu->m_fStepsec);
	printText2D(buf.c_str(),x,y,size);

	y		-= size;
	buf = "Gflops: " + std::to_string(m_pGpu->m_fFlops);
	printText2D(buf.c_str(),x,y,size);

}

void Render::drawALL(Settings cnfg){

	// Ortographic projection
	//glm::mat4 m4Projection	= glm::ortho(-6.0,6.0,-6.0,6.0);
	float l,r,b,t;
	float padding = 0.0;

	l = m_pGpu->m_fk[0].fXmin + padding;
	r = m_pGpu->m_fk[0].fXmax + padding;

	b = m_pGpu->m_fk[0].fYmin + padding;
	t = m_pGpu->m_fk[0].fYmax + padding;

	glm::mat4 m4Projection	= glm::ortho(l,r,b,t);
	float rot[2];
	// Check the rotation
	//int rott = 0;

	switch ( m_rotationType )
	{
	case 0:
		rot[0] = 1.0; rot[1] = 1.0;
		break;
	
	case 1:
		rot[0] = 1.0; rot[1] = -1.0;
		break;
	
	case 2:
		rot[0] = -1.0; rot[1] = 1.0;
		break;
	
	case 3:
		rot[0] = -1.0; rot[1] = -1.0;
		break;
	
	default:
		rot[0] = 1.0; rot[1] = 1.0;
		break;
	}
	
	//cout<<"Rotation tipe: "<<rot[0]<<","<<rot[1]<<endl;
	
	// Drawing
	int numMappings = m_pFrac->getNumOfMaps();
	glUseProgram(m_uiSimpleProgID);
	glUniform1i(m_uNumMappings,numMappings);
	glUniformMatrix4fv(m_uProjection, 1, GL_FALSE, &m4Projection[0][0]);
	glUniform2f(m_uRotation,rot[0],rot[1]);
	glUniform1i(m_uRenderType,m_renderType);
	glUniform2f(m_uRandom,m_randoms[0],m_randoms[1]);
	

	// Draw Fractals
	
	// Draw Fractals
	{
		
		glBindBuffer(GL_ARRAY_BUFFER, m_pGpu->m_fk[0].g_poss);
		glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE, 2 * sizeof(float),(void*)0);
		glBindBuffer(GL_ARRAY_BUFFER, m_pGpu->m_fk[0].g_color);
		glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE, 2 * sizeof(float),(void*)0);
				
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		

		//glPointSize(20.0);
		if(m_renderType == 1 || m_renderType == 3 )	glPointSize(3.0); //Patch configuration
		else 								glPointSize(1.0); //Point configuration	

		glDrawArrays(GL_POINTS,0,m_pFrac->getNumOfPoints());
		//glDrawArrays(GL_POINTS,0,10);
		//float minmaxData[6] = {0,0,0,0,0,0};
		//glGetMinmax(GL_MINMAX, GL_FALSE, GL_RGB, GL_FLOAT, minmaxData);
		//cout.setf(ios::scientific);
		//cout.precision(8);
		//cout << "Min: " << minmaxData[0] << ", " << minmaxData[1] << ", " << minmaxData[2] << endl;
		//cout << "Max: " << minmaxData[3] << ", " << minmaxData[4] << ", " << minmaxData[5] << endl;
		
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		
	}
	
}

void Render::renderToNormal(Settings cnfg){

	// Prepare first window clearing color and enabling OpenGL states/behavior
	glViewport(0, 0, m_uiFboWidth, m_uiFboHeight);
	glClearColor(0.0,0.0,0.0,0.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	drawALL(cnfg);
}
void Render::renderToFBO(Settings cnfg){

	glBindFramebuffer(GL_FRAMEBUFFER, m_uiFboFramBuff);
		// Prepare first window clearing color and enabling OpenGL states/behavior
		glViewport(0, 0, m_uiFboWidth, m_uiFboHeight);
		glClearColor(0.2,0.2,0.2,0.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		drawALL(cnfg);

	glBindFramebuffer(GL_FRAMEBUFFER,0);
////////////////////////////////
	// Draw to a Texture rectangle
	glClearColor(0.0,0.0,0.0,0.0);
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glViewport(0,0,m_uiFboWidth,m_uiFboHeight);

	// Use our shader
	glUseProgram(m_uiTextureProgID);

	// Bind our texture in Texture Unit 0
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_RECTANGLE, m_uiFboTexture);
	// Set our "renderedTexture" sampler to user Texture Unit 0
	glUniform1i(m_uiTextureID, 0);

	// 1rst attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, m_uiQuadVertexBuffer);
	glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,0,(void*)0);

	// Draw the triangles !
	glDrawArrays(GL_TRIANGLES, 0, 6); // 2*3 indices starting at 0 -> 2 triangles
}

void Render::write_paramsto_csv( int numMaps, int count){
    // Make a CSV file with one column of integer values
    // filename - the name of the file
    // colname - the name of the one and only column
    // vals - an integer vector of values
		mapping *m = m_pGpu->m_fk[0].h_map;
		// Write parameters to CSV
		stringstream ss;
		ss<<setw(5)<<setfill('0')<<to_string(count);
		string s = ss.str();
		//cout<<s<<endl;

		string filecsv = m_outPathcsv +"/" + s + ".csv";
    
    // Create an output filestream object
		std::cout.setf(std::ios::scientific);
    std::ofstream myFile(filecsv);
    
		// Send the column name to the stream
    //myFile << colname << "\n";
    
    // Send data to the stream
    for(int i = 0; i < numMaps; ++i)
    {
        myFile<<scientific<< m[i].a <<","
													<< m[i].b <<","
													<< m[i].c <<","
													<< m[i].d <<","
													<< m[i].x <<","
													<< m[i].y <<","
													<< m[i].p <<"\n";
    }
    
		//cout<<"Writting csv..."<<endl;
    // Close the file
    myFile.close();
}
float Render::numPixel(){
	
	int count = 0;
	float density = 0.0f;
	int of = 0;
	int w = m_uiFboWidth;
	int h = m_uiFboHeight;


	//GLubyte *data = m_glFrameBuffer;
	glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, m_glFrameBuffer);


	for(int i=0;i< w;i++){
		of = i*h*3;
		for(int j=0;j<h;j++){
			int r,g,b;
			r = g = b = 0;

			r = (int) m_glFrameBuffer[of + j*3    ];
			g = (int) m_glFrameBuffer[of + j*3 + 1];
			b = (int) m_glFrameBuffer[of + j*3 + 2];

			if( (r != 0 ) || (g != 0) || (b != 0) ){
				//cout<<r<<","<<g<<","<<b<<"  ";
				count++;
			}
		}
		
	}

	density = (float)count/((float)w * (float)h);
	//cout<<endl<<"Total count: "<<count<<"\tDensity: "<<density<<endl;
	//free(data);

	return density;
}

void Render::savePNGfromOpenGLbuffer(int count){
	//const char *filepath = "./data/map.png";
	stringstream ss;
	ss<<setw(5)<<setfill('0')<<to_string(count);
	string s = ss.str();

	string fileimg = m_outPathimg + "/" + s + ".png";
	int width, height;
	width 	= m_uiFboWidth;
	height 	= m_uiFboHeight; 

	#if 0
	GLsizei nrChannels = 3;
	GLsizei stride = nrChannels * width;
	stride += (stride % 4) ? (4 - stride % 4) : 0;
	GLsizei bufferSize = stride * height;
	std::vector<char> buffer(bufferSize);
	glPixelStorei(GL_PACK_ALIGNMENT, 4);
	glReadBuffer(GL_FRONT);
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
	stbi_flip_vertically_on_write(true);
	stbi_write_png(fileimg.c_str(), width, height, nrChannels, buffer.data(), stride);
	#endif
	
	
	GLsizei nrChannels = 3;
	GLsizei stride = nrChannels * width;
	//stride += (stride % 4) ? (4 - stride % 4) : 0;
	//GLsizei bufferSize = stride * height;
	//std::vector<char> buffer(bufferSize);
	//glPixelStorei(GL_PACK_ALIGNMENT, 4);
	//glReadBuffer(GL_FRONT);
	//glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
	stbi_flip_vertically_on_write(true);
	stbi_write_png(fileimg.c_str(), width-2, height-2, nrChannels, m_glFrameBuffer, stride);


	//string fileimg2 = m_outPathimg + "/" + s + ".jpg";
	//stbi_write_jpg(fileimg2.c_str(), width, height, 3, m_glFrameBuffer, 100);

}

void Render::setRootDirParams(string dir, int nclass, float density){
	// For file path
	m_rootDir = dir;
	
	ostringstream o,p;
	o.precision(2);
	o<<fixed<<density;
	p<<fixed<<nclass;
	m_outPathcsv = m_rootDir + "csv_density"+o.str()+"_Class"+p.str();
	m_outPathimg = m_rootDir + "img_density"+o.str()+"_Class"+p.str();

	filesystem::create_directories(m_outPathcsv);
	filesystem::create_directories(m_outPathimg);

}

void Render::setRootDirDataSet(string dir, int nclass, int count,Settings sett){
	// For file path
	m_rootDir = dir;

	//cout<<"Color:"<<cRenderType[s.rt]<<endl;
	//cout<<"Filter:"<<cRenderFilter[s.rf]<<endl<<endl;
	
	if (count>=0){
		stringstream ss;
		ss<<setw(5)<<setfill('0')<<to_string(count);
		string s = ss.str();

		ostringstream p;
		p<<fixed<<nclass;
		m_outPathimg = m_rootDir + "FractalDB-"+p.str()+"_"+ cRenderFilter[sett.rf]+cRenderType[sett.rt]+"/"+s;

		filesystem::create_directory(m_outPathimg);
	}else{
		ostringstream p;
		p<<fixed<<nclass;
		m_outPathimg = m_rootDir + "FractalDB-"+p.str()+"_"+cRenderFilter[sett.rf]+cRenderType[sett.rt];
		filesystem::create_directory(m_outPathimg);
	}
}

void Render::resizeGLbuffer(int w, int h){

	m_uiFboWidth 	= w;
	m_uiFboHeight = h;

	if(m_glFrameBuffer != NULL)
		free(m_glFrameBuffer);
	
	cout<<"Resize render GL buffer "<<endl;
	m_glFrameBuffer = (GLubyte*) malloc (3 * sizeof(GLubyte) * w * h);

}

Render::~Render() {
	std::cout<<"Deleting OpenGL Buffers...\n";
	glDeleteTextures(1, &m_uiFboTexture);
	glDeleteTextures(1, &m_uiFboDepth);
	glDeleteFramebuffers(1, &m_uiFboFramBuff);
	glDeleteVertexArrays(0,&m_uiVertArrayID);

	
	if(m_glFrameBuffer != NULL)
		free(m_glFrameBuffer);

	std::cout<<"Cleaning Text2D...\n";
	cleanupText2D();
}

