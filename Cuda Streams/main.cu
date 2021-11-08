#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>

using namespace std;

__global__ void test_kernel(int step)
{
	printf("Loop: %d\n", step);
}

int main()
{
	int n_stream = 5;
	cudaStream_t* ls_stream;
	ls_stream = (cudaStream_t*)new cudaStream_t[n_stream];

	// Create multiple streams
	for (int i = 0; i < n_stream; i++)
	{
		cudaStreamCreate(&ls_stream[i]);
	}

	// Execute kernel with each CUDA stream
	for (int i = 0; i < n_stream; i++)
	{
		if (i == 3)
		{
			test_kernel << <1, 1, 0, 0 >> > (i); // Default stream is synchronous so no other streams will be launched until completed
		}
		test_kernel << <1, 1, 0, ls_stream[i] >> > (i);
	}

	// Synchronize host and GPU
	cudaDeviceSynchronize();

	// Cleanup
	for (int i = 0; i < n_stream; i++)
	{
		cudaStreamDestroy(ls_stream[i]);
	}
	delete[] ls_stream;
}
