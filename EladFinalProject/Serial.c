#include <stdio.h>
#include <stdlib.h>
#include "proto.h"
#include <time.h>

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

int matching(Object picture, Object obj, int row, int col, double value)
{
    double diff = 0.0, res, size;
    size = (double)(1.0 / (obj.dim * obj.dim));
    double p, o;
    for (int i = 0; i < obj.dim; i++)
    {
        for (int j = 0; j < obj.dim; j++)
        {
            p = (double)picture.members[(picture.dim * (row + i)) + col + j];
            o = (double)obj.members[(i * obj.dim) + j];
            res = myAbs((p - o)/p);
            diff += (res * size);
            if (diff > value)
                return 0; 
        }
        
    }
    return 1;
}

double myAbs(double num)
{
    if (num < 0 )
        return num * -1;
    return num;
}

void resetFindings(int** findings) 
{
    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < DETAILS; j++)
        {
            findings[i][j] = -1;
        }   
    }
}


int main(int argc, char *argv[])
{
    clock_t start = clock();
    double matchingValue;
    int N, K;
    Object *pictures = NULL, *objects = NULL;
    
    FILE* rf = readFile(&matchingValue);
    pictures = readMatrices(rf, &N, 0);
    objects = readMatrices(rf, &K, 1);

    int foundThree = 0, nextPic = 0, nextObj = 0, idx = 0;
    int** findings = (int**)calloc(SIZE, sizeof(int*));
    for (int i = 0; i < SIZE; i++)
        findings[i] = (int*)calloc(DETAILS, sizeof(int));
    FILE* wf = fopen(OUTPUT, "w");

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < K; j++)
        {
            if (nextPic)
                break;
            for (int row = 0; row < pictures[i].dim; row++)
            {
                if (nextPic || nextObj)
                    break;
                for (int col = 0; col < pictures[i].dim; col++)
                {
                    if (row + objects[j].dim < pictures[i].dim && col + objects[j].dim < pictures[i].dim)
                    {
                        if (matching(pictures[i], objects[j], row, col, matchingValue))
                        {
                            foundThree++;
                            findings[idx][0] = objects[j].id;
                            findings[idx][1] = row;
                            findings[idx++][2] = col;
                            nextObj = 1;
                            if (foundThree == 3)
                            {
                                fprintf(wf, "Picture %d: found Objects: ", pictures[i].id);
                                for (int k = 0; k < SIZE-1; k++)
                                {
                                    fprintf(wf, "%d Position(%d,%d) ;", findings[k][0], findings[k][1], findings[k][2]);
                                }
                                fprintf(wf, "%d Position(%d,%d)\n", objects[j].id, row, col);
                                nextPic = 1;
                            }
                            break;
                        }
                    } 
                }
                
            }
            nextObj = 0;
        }
        if (foundThree < 3)
        {
            fprintf(wf, "Picture %d: No three different Objects were found\n", pictures[i].id);   
        }
        foundThree = 0;
        nextPic = 0;
        resetFindings(findings);
        idx = 0;
    }
    fclose(wf);
    clock_t end = clock();
    double sec = (double)((end - start) / CLOCKS_PER_SEC);
    printf("Execution time: %lfs\n", sec);
    return 0;
}
