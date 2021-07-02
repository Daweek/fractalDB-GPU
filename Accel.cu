/*
 * Accel.cu
 *
 *  Created on: May 31, 2021
 *      Author: Edg@r j.
 */
#include "Accel.cuh"

__global__ void find_borders_kernel(float2* array, float4 *brd, int *mutex, unsigned int n)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;

	__shared__ float cacheMaxX[256];
	__shared__ float cacheMaxY[256];
	__shared__ float cacheMinX[256];
	__shared__ float cacheMinY[256];


	float maxX = 1.0;
	float maxY = 1.0;
	float minX = -1.0;
	float minY = -1.0;


	while(index + offset < n){
		maxX = fmaxf(maxX, array[index + offset].x);
		maxY = fmaxf(maxY, array[index + offset].y);

		minX = fminf(minX, array[index + offset].x);
		minY = fminf(minY, array[index + offset].y);

		offset += stride;
	}

	cacheMaxX[threadIdx.x] = maxX;
	cacheMaxY[threadIdx.x] = maxY;
	cacheMinX[threadIdx.x] = minX;
	cacheMinY[threadIdx.x] = minY;

	__syncthreads();


	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cacheMaxX[threadIdx.x] = fmaxf(cacheMaxX[threadIdx.x], cacheMaxX[threadIdx.x + i]);
			cacheMaxY[threadIdx.x] = fmaxf(cacheMaxY[threadIdx.x], cacheMaxY[threadIdx.x + i]);
			cacheMinX[threadIdx.x] = fminf(cacheMinX[threadIdx.x], cacheMinX[threadIdx.x + i]);
			cacheMinY[threadIdx.x] = fminf(cacheMinY[threadIdx.x], cacheMinY[threadIdx.x + i]);
		}

		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0){
		while(atomicCAS(mutex,0,1) != 0);  //lock
		brd[0].x = fmaxf(brd[0].x, cacheMaxX[0]);
		brd[0].y = fmaxf(brd[0].y, cacheMaxY[0]);
		brd[0].z = fminf(brd[0].z, cacheMinX[0]);
		brd[0].w = fminf(brd[0].w, cacheMinY[0]);
		atomicExch(mutex, 0);  //unlock
	}
}

__global__ void kernel_2(float2* d_poss,float2* d_color , int numPoints,mapping *d_mappings, int numMappings)
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
    d_poss[i].x = currentPosition.x ;
    d_poss[i].y = currentPosition.y ;

    // set the iteration percentage and current target mapping
    d_color[i].x =  i / (float) numPoints;
    d_color[i].y = currentTarget;

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



__global__ void kernel_test(float2* d_pointData, int numPoints,mapping *d_mappings, int numMappings)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  //int stride = blockDim.x * gridDim.x;

  int currentTarget = index % numMappings;

  //d_pointData[index].x = 0.0f + currentTarget * 0.5f;
  d_pointData[index].y = 0.0f + currentTarget * 0.10f;
  //d_pointData[index].x = 0.0f;
  //d_pointData[index].y = 0.0f;

}


//__global__ void kernel_P10(float2 d_poss[MAPS_GPU][MAX_POINTS],float2 d_color[MAPS_GPU][MAX_POINTS], int numPoints, mapping *d_mappings, int numMappings)
__global__ void kernel_P10(float2 d_poss[MAPS_GPU][MAX_POINTS],float2* d_color, int numPoints, mapping *d_mappings, int numMappings)


{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
	int blockID = blockIdx.x;

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
    d_poss[0][i].x = currentPosition.x ;
    d_poss[0][i].y = currentPosition.y ;

    // set the iteration percentage and current target mapping
    d_color[i].x =  i / (float) numPoints;
    d_color[i].y = currentTarget;

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

__global__ void kernel_test_P(float2* d_poss, int numPoints,mapping *d_mappings, int numMappings)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  //int stride = blockDim.x * gridDim.x;

  int currentTarget = index % numMappings;

  //d_pointData[index].x = 0.0f + currentTarget * 0.5f;
  d_poss[index].y = 0.0f + currentTarget * 0.10f;
  //d_pointData[index].x = 0.0f;
  //d_pointData[index].y = 0.0f;

}


Accel::Accel() {

	// Initialize CUDA
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaGetDevice(&m_cuDevice));
	checkCudaErrors(cudaGetDeviceProperties(&m_cuDevProp,m_cuDevice));

	cudaDriverGetVersion(&m_driverVersion);
	cudaRuntimeGetVersion(&m_runtimeVersion);

	// Print device properties
	printf("............................GPU...........................\t\n");
	printf("\tDevice Name: %s\n", m_cuDevProp.name);
	printf("\tCUDA Driver Version / Runtime Version: %d.%d / %d.%d\n",
					m_driverVersion / 1000, (m_driverVersion % 100) / 10,
					m_runtimeVersion / 1000, (m_runtimeVersion % 100) / 10);
	printf("\tCompute Capability: %d.%d\n", m_cuDevProp.major, m_cuDevProp.minor);
	printf("\tTotal Global Memory: %ld bytes\n", m_cuDevProp.totalGlobalMem);
	printf("\tNumber of Multiprocessors: %d\n", m_cuDevProp.multiProcessorCount);
	printf("\tMaximum Threads per Multiprocessor: %d\n",
		m_cuDevProp.maxThreadsPerMultiProcessor);
	printf("\tTotal Number of Threads: %d\n", m_cuDevProp.multiProcessorCount *
		m_cuDevProp.maxThreadsPerMultiProcessor);
	printf("\tMaximum Threads per Block: %d\n", m_cuDevProp.maxThreadsPerBlock);
	printf(".........................................................\t\n\n");
	

	// Setting up all pointers
	//d_glmap = d_glPoss = d_glColor = NULL;
	//d_glPoss = d_glColor = NULL;
	//d_map		= NULL;
	//g_strucMapVBO		= NULL;
	//d_borders = NULL;
	//h_borders = NULL;
	//d_mutex = NULL;
	
	// CUDA related structs
	//g_mapVBO = 0;

	// Timer related
	m_fFlops = m_fStepsec = 0.0f;

	// Memory Flags related
	m_bChangeInterop = m_bChangeMalloc = true;

	m_numBlocks = m_blockSize = 0;
	//m_fXmax = m_fXmin = m_fYmax = m_fYmin = 0.0;
}

void Accel::interopCUDA(){
	std::cout<<"Seting up CUDA-OpenGL buffer...\n\n";
  // Prepare graphics interoperability



  //if(g_strucMapVBO != NULL) 
	//	checkCudaErrors(cudaGraphicsUnregisterResource(g_strucMapVBO));

	for(int p=0; p<MAPS_GPU; p++){
		if(g_strucPoss[p] != NULL) 
			checkCudaErrors(cudaGraphicsUnregisterResource(g_strucPoss[p]));

		if(g_strucColor[p] != NULL) 
			checkCudaErrors(cudaGraphicsUnregisterResource(g_strucColor[p]));


		//glDeleteBuffers(1,&g_mapVBO);
		glDeleteBuffers(1,&g_poss[p]);
		glDeleteBuffers(1,&g_color[p]);
	

		glGenBuffers(1, &g_poss[p]);
		glBindBuffer(GL_ARRAY_BUFFER, g_poss[p]);
		unsigned int sizeP = MAX_POINTS * 2 * sizeof(float);
		glBufferData(GL_ARRAY_BUFFER, sizeP, 0, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glGenBuffers(1, &g_color[p]);
		glBindBuffer(GL_ARRAY_BUFFER, g_color[p]);
		unsigned int sizeC = MAX_POINTS * 2 * sizeof(float);
		glBufferData(GL_ARRAY_BUFFER, sizeC, 0, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// Register CUDA and OpenGL Interop
		//checkCudaErrors(cudaGraphicsGLRegisterBuffer(&g_strucMapVBO,g_mapVBO,cudaGraphicsMapFlagsNone));
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&g_strucPoss[p],g_poss[p],cudaGraphicsMapFlagsNone));
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(&g_strucColor[p],g_color[p],cudaGraphicsMapFlagsNone));
	}

	// Creation of share buffer between CUDA and OpenGL
  // For mapping position and color
  //glGenBuffers(1, &g_mapVBO);
  //glBindBuffer(GL_ARRAY_BUFFER, g_mapVBO);
  //unsigned int size = MAX_POINTS * 4 * sizeof(float);
  //glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
  //glBindBuffer(GL_ARRAY_BUFFER, 0);


}


//void Accel::malloCUDA(mapping *mapped[MAPS_GPU], int numMaps[]){
void Accel::malloCUDA(std::vector<mapping*>& mapped, int numMaps[]){

	// For the parameters of the Fractals

	
		if(d_map[k] != NULL)
			checkCudaErrors(cudaFree(d_map[k]));
  	checkCudaErrors(cudaMalloc((void**)&d_map[k],numMaps[k]*sizeof(mapping)));
  	checkCudaErrors(cudaMemcpy(d_map[k],mapped[k],numMaps[k]*sizeof(mapping),cudaMemcpyHostToDevice));
	
	for(int k=0; k<MAPS_GPU ; k++){
		// To check borders
		if(d_borders[k] != NULL)
			checkCudaErrors(cudaFree(d_borders[k]));
		cudaMalloc((void**)&d_borders[k],sizeof(float4));
		cudaMemset(d_borders[k],0, sizeof(float4));

		if(d_mutex[k] != NULL)
			checkCudaErrors(cudaFree(d_mutex[k]));
		cudaMalloc((void**)&d_mutex[k],sizeof(int));
		cudaMemset(d_mutex[k], 0, sizeof(int));

		if(h_borders[k] != NULL)
			free(h_borders[k]);
		h_borders[k] = (float*)malloc(4*sizeof(float)); 
	}
}

void Accel::fractalKernel(int numMappings[], int numPoints){

	m_numBlocks = 1;
	m_blockSize	= 1024;

  size_t mapsizevbo;
  
	for(int p=0; p<MAPS_GPU; p++){
		checkCudaErrors(cudaGraphicsMapResources(1,&g_strucPoss[p],0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_glPoss[p],&mapsizevbo,g_strucPoss[p]));

		checkCudaErrors(cudaGraphicsMapResources(1,&g_strucColor[p],0));
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_glColor[p],&mapsizevbo,g_strucColor[p]));
	}

  cudaEvent_t start, stop;
  checkCudaErrors( cudaEventCreate(&start) );
  checkCudaErrors( cudaEventCreate(&stop) );
  
  checkCudaErrors( cudaEventRecord(start) );
  
	
	// Compute Fractal
		//kernel_2<<<m_numBlocks, m_blockSize, numMappings * sizeof(mapping)>>>
    	//((float2*)d_glPoss[0],(float2*)d_glColor[0] , numPoints, d_map[0], numMappings);


		//kernel_P10<<<m_numBlocks, m_blockSize, numMappings * sizeof(mapping)>>>
		//	(*reinterpret_cast<float2 (*)[MAPS_GPU][MAX_POINTS]>(&d_glPoss),(float2 (*))d_glColor , numPoints, d_map[0], numMappings);

		kernel_test_P<<<m_numBlocks, m_blockSize, numMappings[0] * sizeof(mapping)>>>
			((float2 (*))d_glPoss[0], numPoints,(mapping (*)) d_map[0], numMappings[0]);
			
		checkCudaErrors( cudaPeekAtLastError() );
		checkCudaErrors( cudaDeviceSynchronize() );
		
	// Compute Borders of the fractal
	
	dim3 gridSize = 256;
	dim3 blockSize = 256;

	cudaMemset(d_mutex[0], 0, sizeof(int));

  
		find_borders_kernel<<< gridSize, blockSize >>>
			((float2*)d_glPoss[0],d_borders[0], d_mutex[0], (unsigned int)numPoints);
		   
	checkCudaErrors(cudaMemcpy(h_borders[0], d_borders[0], sizeof(float4), cudaMemcpyDeviceToHost));

	m_fXmax[0] = h_borders[0][0];
	m_fYmax[0] = h_borders[0][1];
	m_fXmin[0] = h_borders[0][2];
	m_fYmin[0] = h_borders[0][3];
	
	/*
		cout<<"Maximum X found on gpu was: "<<m_fXmax<<endl;
		cout<<"Maximum Y found on gpu was: "<<m_fYmax<<endl;
		cout<<"Minimum X found on gpu was: "<<m_fXmin<<endl;
		cout<<"Minimum Y found on gpu was: "<<m_fYmin<<endl<<endl;
	*/	

  checkCudaErrors( cudaEventRecord(stop) );

  // handle any synchronous and asynchronous kernel errors
  checkCudaErrors( cudaGetLastError() );
  checkCudaErrors( cudaDeviceSynchronize() );

  // record and print kernel timing
  checkCudaErrors( cudaEventSynchronize(stop) );
  m_kernel_mili = 0;
  checkCudaErrors( cudaEventElapsedTime(&m_kernel_mili, start, stop) );

  // Unmap OpenGL resources
	for(int p=0; p<MAPS_GPU; p++){
  	checkCudaErrors(cudaGraphicsUnmapResources(1,&g_strucPoss[p],0));
		checkCudaErrors(cudaGraphicsUnmapResources(1,&g_strucColor[p],0));
	}
	
}

Accel::~Accel() {
	// Unregister if CUDA-InteropGL
	std::cout<<"Unregistering CUDA-GL Resources...\n";

	for(int k=0; k<MAPS_GPU ; k++){
		if(g_strucPoss[k] != NULL) 
			checkCudaErrors(cudaGraphicsUnregisterResource(g_strucPoss[k]));

		if(g_strucColor != NULL) 
			checkCudaErrors(cudaGraphicsUnregisterResource(g_strucColor[k]));
		// Free CUDA memory
	
		if(d_map[k] != NULL)
			checkCudaErrors(cudaFree(d_map[k]));
  	
		// To check borders
		if(d_borders[k] != NULL)
			checkCudaErrors(cudaFree(d_borders[k]));
	
		if(d_mutex[k] != NULL)
			checkCudaErrors(cudaFree(d_mutex[k]));
	
		if(h_borders[k] != NULL)
			free(h_borders[k]);
	
	}
}

