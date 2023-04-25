#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "CudaFunctions.cuh"
#include "proto.h"


__host__ void allocatePictureOnGPU(Object picture, int** devicePicture)
{
    int pixelsInPicture = picture.dim * picture.dim;

    cudaError_t error = cudaSuccess;
    error = cudaMalloc(devicePicture, pixelsInPicture * sizeof(int));
    if (error != cudaSuccess)
    {
        printf("Cannot allocate GPU memory for image: %s (%d)\n", cudaGetErrorString(error), error);
    	exit(0);
    }
    error = cudaMemcpy(*devicePicture, picture.members, pixelsInPicture * sizeof(int), cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("Cannot copy image to GPU: %s (%d)\n", cudaGetErrorString(error), error);
    	exit(0);
    }
}

__host__ void freePictureOnGPU(int** devicePicture)
{
    cudaError_t error = cudaSuccess;

    error = cudaFree(*devicePicture);
    if (error != cudaSuccess)
    {
        printf("Cannot free image from GPU: %s (%d)\n", cudaGetErrorString(error), error);
    	exit(0);
    }
}

__host__ double* searchObjectInPictureOnGPU(Object object, int* devicePicture, int pictureDim)
{
    int positionPerDim = positionsFlags(pictureDim, object.dim);
    int positionCount = positionPerDim * positionPerDim;
    int blockLen = roundUpPowerOf2(object.dim);
    int *deviceObject;
    double *hostMatchings, *deviceMatchings;

    cudaError_t error = cudaSuccess;
    hostMatchings = (double*)calloc(positionCount, sizeof(double));
    if (hostMatchings == NULL) 
	{
		printf("Cannot allocate meory for position flags array\n");
		exit(0);
	}

    allocatePictureOnGPU(object, &deviceObject);

    error = cudaMalloc(&deviceMatchings, positionCount * sizeof(double));
    if (error != cudaSuccess)
    {
        printf("Cannot allocate GPU memory for matchings array: %s (%d)\n", cudaGetErrorString(error), error);
    	exit(0);
    }

    error = cudaMemset(deviceMatchings, 0, positionCount * sizeof(double));
    if (error != cudaSuccess)
    {
        printf("Cannot initialize matchings array on GPU: %s (%d)\n", cudaGetErrorString(error), error);
    	exit(0);
    }

    blockLen = blockLen < BLOCKDIMENSION? blockLen: BLOCKDIMENSION;

    dim3 gridDimensions(positionPerDim, positionPerDim);
    dim3 blockDimensions(blockLen, blockLen);

    // ======================== call for search ======================
    searchMatching<<<gridDimensions, blockDimensions>>>(devicePicture, pictureDim, deviceObject, object.dim, deviceMatchings);

    error = cudaMemcpy(hostMatchings, deviceMatchings, positionCount * sizeof(double), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        printf("Cannot copy position flags from GPU to host: %s (%d)\n", cudaGetErrorString(error), error);
    	exit(0);
    }

    error = cudaFree(deviceMatchings);
    if (error != cudaSuccess)
    {
        printf("Cannot free matchings array from GPU: %s (%d)\n", cudaGetErrorString(error), error);
    	exit(0);
    }

    freePictureOnGPU(&deviceObject);

    return hostMatchings;
}

__host__ __device__ int positionsFlags(int pictureDim, int ObjectDim)
{
    return pictureDim - ObjectDim + 1;
}

__global__ void searchMatching(int* devicePicture, int pictureDim, int* deviceObject, int objectDim, double* deviceMatchings)
{
    __shared__ double differencesArray[THREADSPERBLOCK];

    if (blockIdx.x >= pictureDim || blockIdx.y >= pictureDim)
        return;

    int tid = threadIdx.x * blockDim.y + threadIdx.y;
    int bid = blockIdx.x * gridDim.y + blockIdx.y;

    int pixelsPerThreadX = ((objectDim - 1) / blockDim.x) + 1;
    int pixelsPerThreadY = ((objectDim - 1) / blockDim.y) + 1;

    int startX = threadIdx.x * pixelsPerThreadX;
    int startY = threadIdx.y * pixelsPerThreadY;

    int endX = MIN(startX + pixelsPerThreadX, objectDim);
    int endY = MIN(startY + pixelsPerThreadY, objectDim);

    double diff = 0.0;
    int* picture = devicePicture + (blockIdx.x * pictureDim) + blockIdx.y;

    for (int row = startX; row < endX; row++)
    {
        for (int col = startY; col < endY; col++)
        {
            diff += difference(
                picture[(row * pictureDim) + col],
                deviceObject[(row * objectDim) + col]
            );
        }
    }
    
    differencesArray[tid] = diff;
    __syncthreads();

    for (int step = (blockDim.x * blockDim.y) / 2; step > 0; step /= 2)
    {
        if (tid < step)
            differencesArray[tid] += differencesArray[tid + step];
        
        __syncthreads();
    }
    
    if (tid == 0)
        deviceMatchings[bid] = differencesArray[0];

}

__device__ double difference(int p, int o)
{
    double pd = (double)p;
    double od = (double)o;
    return abs((pd -od) / pd);
}

int roundUpPowerOf2(int val)
{
	val--;
    val |= val >> 1;
    val |= val >> 2;
    val |= val >> 4;
    val |= val >> 8;
    val |= val >> 16;
    val++;

    return val;
}

