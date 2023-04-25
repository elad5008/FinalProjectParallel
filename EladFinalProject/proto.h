#pragma once
#include <stdio.h>

#define INPUT "input.txt"
#define OUTPUT "output.txt"

#define EMPTY -1

#define SIZE 3

// Objct & Picture struct
typedef struct {
    int id;
    int dim;
    int * members;
} Object;

typedef struct {
    int picId;
    int objsId[SIZE];
    int objsRows[SIZE];
    int objsCols[SIZE];
} Findings;


FILE* readFile(double* value);
Object readMat(FILE* fp);
Object* readMatrices(FILE* fp, int* size, int toClose) ;


Findings* searchObjectsInPictures(Object* pictures, int numOfPics, Object* objects, int numOfObjects, double matchingValue);
Findings searchObjectsInPicture(Object picture, Object* objects, int numOfObjects, double matchingValue);
int* searchObjectInPicture(int pictureDim, int* devicePicture, Object object, double matchingValue);

