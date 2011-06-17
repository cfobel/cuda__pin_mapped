// incrementMappedArrayInPlace.cu
#include <iostream>
#include <cstdio>
#include <cassert>
#include <boost/format.hpp>
#include <cuda.h>
using namespace std;

#define _ boost::format
 
// define the problem and block size
#define NUMBER_OF_ARRAY_ELEMENTS 100000
#define N_THREADS_PER_BLOCK 256
 
void incrementArrayOnHost(float *a, int N)
{
  int i;
  for (i=0; i < N; i++) a[i] = a[i]+1.f;
}
 
__global__ void incrementArrayOnDevice(float *a, int N)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) a[idx] = a[idx]+1.f;
}
 
void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
    exit(EXIT_FAILURE);
  }                        
}
 

int main(int argc, char** argv) {
	int major = 0;
    int minor = 0;
	int deviceCount = 0;

	CUresult err = cuInit(0);
    cuDeviceGetCount(&deviceCount);

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
        return 0;
	}

    // This function call returns 9999 for both major & minor fields, 
    // if no CUDA capable devices are present
    cuDeviceComputeCapability(&major, &minor, 0);


    float *a_m; // pointer to host memory
    float *a_d; // pointer to mapped device memory
    float *check_h;   // pointer to host memory used to check results
    int i, N = NUMBER_OF_ARRAY_ELEMENTS;
    size_t size = N*sizeof(float);
    cudaDeviceProp deviceProp;
    
    #if CUDART_VERSION < 2020
    #error "This CUDART version does not support mapped memory!\n"
    #endif
    
    // Get properties and verify device 0 supports mapped memory
    cudaGetDeviceProperties(&deviceProp, 0);
    checkCUDAError("cudaGetDeviceProperties");
    
    if(!deviceProp.canMapHostMemory) {
        fprintf(stderr, "Device %d cannot map host memory!\n", 0);
        exit(EXIT_FAILURE);
    }
    
    // set the device flags for mapping host memory
    cudaSetDeviceFlags(cudaDeviceMapHost);
    checkCUDAError("cudaSetDeviceFlags");
    
    // allocate mapped arrays
    cout << "allocate mapped arrays...";
    cout.flush();
    cudaHostAlloc((void **)&a_m, size, cudaHostAllocMapped);
    checkCUDAError("cudaHostAllocMapped");
    cout << "DONE" << endl;
    
    // Get the device pointers to the mapped memory
    cout << "Get the device pointers to the mapped memory...";
    cout.flush();
    cudaHostGetDevicePointer((void **)&a_d, (void *)a_m, 0);
    checkCUDAError("cudaHostGetDevicePointer");
    cout << "DONE" << endl;
    
    // initialization of host data
    cout << "Initialize host data...";
    cout.flush();
    for (i=0; i<N; i++) a_m[i] = (float)i;
    cout << "DONE" << endl;
    
    // do calculation on device:
    // Part 1 of 2. Compute execution configuration
    int blockSize = N_THREADS_PER_BLOCK;
    int nBlocks = N/blockSize + (N%blockSize > 0?1:0);
    
    // Part 2 of 2. Call incrementArrayOnDevice kernel
    cout << "Request calculation on GPU...";
    cout.flush();
    incrementArrayOnDevice <<< nBlocks, blockSize >>> (a_d, N);
    checkCUDAError("incrementArrayOnDevice");
    cout << "DONE" << endl;
    
    /* Note the allocation, initialization and call to incrementArrayOnHost
        occurs asynchronously to the GPU */
    cout << "Perform calculation on CPU...";
    cout.flush();
    check_h = (float *)malloc(size);
    for (i=0; i<N; i++) check_h[i] = (float)i;
    incrementArrayOnHost(check_h, N);
    cout << "DONE" << endl;
    
    // Make certain that all threads are idle before proceeding
    cout << "Wait for GPU calculation to finish...";
    cout.flush();
    cudaThreadSynchronize();
    checkCUDAError("cudaThreadSynchronize");
    cout << "DONE" << endl;
    
    // check results
    cout << "Verify results...";
    cout.flush();
    for (i=0; i<N; i++) assert(check_h[i] == a_m[i]);
    cout << "DONE" << endl;
    
    // cleanup
    free(check_h); // free host memory
    cudaFreeHost(a_m); // free mapped memory (and device pointers)
}
