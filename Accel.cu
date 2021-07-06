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

	// Start the Fractal structure
	m_fk[0].g_strucPoss = NULL;	
	m_fk[0].g_strucColor = NULL;
	m_fk[0].d_glPoss = NULL;
	m_fk[0].d_glColor = NULL;
	
	m_fk[0].d_borders = NULL;
	m_fk[0].h_borders = NULL;

	m_fk[0].fXmin = 0.0;
	m_fk[0].fXmax = 0.0;
	m_fk[0].fYmin = 0.0;
	m_fk[0].fYmax = 0.0;

	m_fk[0].d_mutex = NULL;

	m_fk[0].d_map = NULL;
	m_fk[0].h_map	= NULL;

	// Setting up all pointers
	//d_glColor = NULL;
	//d_glPoss = NULL;
	//d_map		= NULL;
	//d_borders = NULL;
	//h_borders = NULL;
	//d_mutex = NULL;
		
	// CUDA related structs
	//g_strucColor  = NULL;
	//g_strucPoss = NULL;

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

	if(m_fk[0].g_strucPoss != NULL) 
		checkCudaErrors(cudaGraphicsUnregisterResource(m_fk[0].g_strucPoss));

	if(m_fk[0].g_strucColor != NULL) 
		checkCudaErrors(cudaGraphicsUnregisterResource(m_fk[0].g_strucColor));

	glDeleteBuffers(1,&m_fk[0].g_poss);
	glDeleteBuffers(1,&m_fk[0].g_color);

  // Creation of share buffer between CUDA and OpenGL

	glGenBuffers(1, &m_fk[0].g_poss);
  glBindBuffer(GL_ARRAY_BUFFER, m_fk[0].g_poss);
  unsigned int sizeP = MAX_POINTS * 2 * sizeof(float);
  glBufferData(GL_ARRAY_BUFFER, sizeP, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &m_fk[0].g_color);
  glBindBuffer(GL_ARRAY_BUFFER, m_fk[0].g_color);
  unsigned int sizeC = MAX_POINTS * 2 * sizeof(float);
  glBufferData(GL_ARRAY_BUFFER, sizeC, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Register CUDA and OpenGL Interop
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_fk[0].g_strucPoss,m_fk[0].g_poss,cudaGraphicsMapFlagsNone));
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_fk[0].g_strucColor,m_fk[0].g_color,cudaGraphicsMapFlagsNone));

}


void Accel::malloCUDA(int numMaps){

	// For params of fractals
	if(m_fk[0].d_map != NULL)
		checkCudaErrors(cudaFree(m_fk[0].d_map));
  checkCudaErrors(cudaMalloc((void**)&m_fk[0].d_map,numMaps*sizeof(mapping)));
  checkCudaErrors(cudaMemcpy(m_fk[0].d_map,m_fk[0].h_map,numMaps*sizeof(mapping),cudaMemcpyHostToDevice));

	// To check borders
	if(m_fk[0].d_borders != NULL)
		checkCudaErrors(cudaFree(m_fk[0].d_borders));
	cudaMalloc((void**)&m_fk[0].d_borders,sizeof(float4));
	cudaMemset(m_fk[0].d_borders,0, sizeof(float4));

	if(m_fk[0].d_mutex != NULL)
		checkCudaErrors(cudaFree(m_fk[0].d_mutex));
	cudaMalloc((void**)&m_fk[0].d_mutex,sizeof(int));
	cudaMemset(m_fk[0].d_mutex, 0, sizeof(int));


	if(m_fk[0].h_borders != NULL)
		free(m_fk[0].h_borders);
	m_fk[0].h_borders = (float*)malloc(4*sizeof(float)); 

}

void Accel::fractalKernel(int numMappings, int numPoints){

	m_numBlocks = 1;
	m_blockSize	= 1024;

  size_t mapsizevbo;

  checkCudaErrors(cudaGraphicsMapResources(1,&m_fk[0].g_strucPoss,0));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_fk[0].d_glPoss,&mapsizevbo,m_fk[0].g_strucPoss));

	checkCudaErrors(cudaGraphicsMapResources(1,&m_fk[0].g_strucColor,0));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&m_fk[0].d_glColor,&mapsizevbo,m_fk[0].g_strucColor));
  
  cudaEvent_t start, stop;
  checkCudaErrors( cudaEventCreate(&start) );
  checkCudaErrors( cudaEventCreate(&stop) );
  
  checkCudaErrors( cudaEventRecord(start) );
  
	
	// Compute Fractal
		kernel_2<<<m_numBlocks, m_blockSize, numMappings * sizeof(mapping)>>>
    	((float2*)m_fk[0].d_glPoss,(float2*)m_fk[0].d_glColor , numPoints, m_fk[0].d_map, numMappings);
			
			
		
	// Compute Borders of the fractal
	
	dim3 gridSize = 256;
	dim3 blockSize = 256;

	cudaMemset(m_fk[0].d_mutex, 0, sizeof(int));

  
		find_borders_kernel<<< gridSize, blockSize >>>
			((float2*)m_fk[0].d_glPoss,m_fk[0].d_borders, m_fk[0].d_mutex, (unsigned int)numPoints);
		   
	checkCudaErrors(cudaMemcpy(m_fk[0].h_borders, m_fk[0].d_borders, sizeof(float4), cudaMemcpyDeviceToHost));

	m_fk[0].fXmax = m_fk[0].h_borders[0];
	m_fk[0].fYmax = m_fk[0].h_borders[1];
	m_fk[0].fXmin = m_fk[0].h_borders[2];
	m_fk[0].fYmin = m_fk[0].h_borders[3];
	
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
	checkCudaErrors(cudaGraphicsUnmapResources(1,&m_fk[0].g_strucPoss,0));
	checkCudaErrors(cudaGraphicsUnmapResources(1,&m_fk[0].g_strucColor,0));
	
}

Accel::~Accel() {
	// Unregister if CUDA-InteropGL
	std::cout<<"Unregistering CUDA-GL Resources...\n";

	if(m_fk[0].g_strucPoss != NULL) 
		checkCudaErrors(cudaGraphicsUnregisterResource(m_fk[0].g_strucPoss));

	if(m_fk[0].g_strucColor != NULL) 
		checkCudaErrors(cudaGraphicsUnregisterResource(m_fk[0].g_strucColor));
	
	if(m_fk[0].h_borders != NULL)
		free(m_fk[0].h_borders);

	if(m_fk[0].d_map != NULL)
		checkCudaErrors(cudaFree(m_fk[0].d_map));
	// Free memory for HALF interop
	//delete [] m_fPossVBO;
}

