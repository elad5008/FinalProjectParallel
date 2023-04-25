#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "CudaFunctions.cuh"
#include "proto.h"

FILE* readFile(double* value) 
{
    FILE* fp = fopen(INPUT, "r");
    if (!fp)
        return NULL;
    fscanf(fp, "%lf\n", value);
    return fp;
}

Object readMat(FILE* fp)
{
    Object o;
    fscanf(fp, "%d\n", &o.id);
    fscanf(fp, "%d\n", &o.dim);
    int pixelsSize = o.dim * o.dim;
    o.members = (int*)calloc(pixelsSize, sizeof(int));
    for (int i = 0; i < pixelsSize; i++)
        fscanf(fp, "%d", &o.members[i]);
    
    return o;
}

Object* readMatrices(FILE* fp, int* size, int toClose) 
{
    fscanf(fp, "%d\n", size);
    Object* objects = (Object*)calloc(*size, sizeof(Object));
    if (!objects) 
    {
        printf("problem in memory allocation!\n");
        exit(0);
        return NULL;
    }
    for (int i = 0; i < *size; i++)
        objects[i] = readMat(fp);
    
    if (toClose)
        fclose(fp);
    
    return objects;
}

Findings* searchObjectsInPictures(Object* pictures, int numOfPics, Object* objects, int numOfObjects, double matchingValue)
{
    Findings* results = (Findings*)calloc(numOfPics, sizeof(Findings));

    #pragma omp parallel for shared(results)
    for (int i = 0; i < numOfPics; i++)
    {
        results[i] = searchObjectsInPicture(pictures[i], objects, numOfObjects, matchingValue);
    }
    
    return results;
}

Findings searchObjectsInPicture(Object picture, Object* objects, int numOfObjects, double matchingValue)
{
    Findings result;
    int* devicePicture = NULL;
    int idx = 0;

    result.picId = EMPTY;
    for (int j = 0; j < SIZE; j++)
    {
        result.objsId[j] = EMPTY;
        result.objsRows[j] = EMPTY;
        result.objsCols[j] = EMPTY;
    } 

    allocatePictureOnGPU(picture, &devicePicture);
    int* objectData = NULL;
    for (int i = 0; i < numOfObjects; i++)
    {
        objectData = searchObjectInPicture(picture.dim, devicePicture, objects[i], matchingValue);
        if (objectData[0])
        {
            result.objsId[idx] = objectData[1];
            result.objsRows[idx] = objectData[2];
            result.objsCols[idx++] = objectData[3];
        }
        if (idx == 3)
        {
            result.picId = picture.id;
            break;
        }
    }

    freePictureOnGPU(&devicePicture);

    if (idx < 3)
    {
        for (int j = 0; j < SIZE; j++)
        {
            result.objsId[j] = EMPTY;
            result.objsRows[j] = EMPTY;
            result.objsCols[j] = EMPTY;
        } 
    }

    return result;
}

int* searchObjectInPicture(int pictureDim, int* devicePicture, Object object, double matchingValue)
{
    int* objectData = (int*)calloc(SIZE+1, sizeof(int));
    objectData[0] = 0;
    objectData[1] = EMPTY;
    objectData[2] = EMPTY;
    objectData[3] = EMPTY;
    double *matchings;

    matchings = searchObjectInPictureOnGPU(object, devicePicture, pictureDim);
    int positionFlagsPerDim = positionsFlags(pictureDim, object.dim), idx = 0;
    int toBreak = 0, numOfCalcs = object.dim * object.dim;

    for (int row = 0; row < positionFlagsPerDim; row++)
    {
        if (toBreak)
            break;
        for (int col = 0; col < positionFlagsPerDim; col++, idx++)
        {
            if ((matchings[idx] / numOfCalcs) < matchingValue)
            {
                objectData[0] = 1;
                objectData[1] = object.id;
                objectData[2] = row;
                objectData[3] = col;
                toBreak = 1;
                break;
            }
        }
    }

    return objectData;
}

