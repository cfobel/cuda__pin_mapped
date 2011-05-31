#include <cuda.h>

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
    while(1);

    return 0;
}
