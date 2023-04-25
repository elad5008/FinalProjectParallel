#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "proto.h"

#define MIN(x, y) x < y? x: y;

#define BLOCKDIMENSION 32					// for CUDA calculations
#define THREADSPERBLOCK BLOCKDIMENSION * BLOCKDIMENSION 	// for CUDA calculations

// ==================== host functions ====================
__host__ void allocatePictureOnGPU(Object picture, int** devicePicture);
__host__ void freePictureOnGPU(int** devicePicture);
__host__ double* searchObjectInPictureOnGPU(Object object, int* devicePicture, int pictureDim);

// ==================== device function ====================
__device__ double difference(int p, int o);

// ==================== host & device functions ====================
__host__ __device__ int positionsFlags(int pictureDim, int ObjectDim);
int roundUpPowerOf2(int val);

// // ==================== kernel function ====================
__global__ void searchMatching(int* devicePicture, int pictureDim, int* deviceObject, int objectDim, double* deviceMatchings);
