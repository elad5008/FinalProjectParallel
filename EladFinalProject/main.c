#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "mpi.h"
#include "proto.h"

int main(int argc, char *argv[])
{
    int rank, size;
    double matchingValue, start;
    int N, K;
    MPI_Status status;
    Object *pictures = NULL, *objects = NULL;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2)
    {
        printf("The program can run only with 2 proccesses\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    if (rank == 0)
    {
        start = MPI_Wtime();
        FILE* rf = readFile(&matchingValue);
        if (!rf)
        {
            printf("Error in opening input file\n");
            MPI_Abort(MPI_COMM_WORLD, 0);
        }
        pictures = readMatrices(rf, &N, 0);
        objects = readMatrices(rf, &K, 1);
    }

    MPI_Bcast(&matchingValue, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    Object *myPics;
    int numOfPics = N/size;
    myPics = (Object*)calloc(numOfPics, sizeof(Object));

    if (rank == 0)
    {
        if (N % size)
        {
            numOfPics++;
            free(myPics);
            myPics = (Object*)calloc(numOfPics, sizeof(Object));
        }

        for (int i = 0; i < numOfPics; i++)
            myPics[i] = pictures[i];
        
        for (int i = numOfPics; i < N; i++)
        {
            MPI_Send(&pictures[i].id, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Send(&pictures[i].dim, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            int pixels = pictures[i].dim * pictures[i].dim;
            MPI_Send(pictures[i].members, pixels, MPI_INT, 1, 0, MPI_COMM_WORLD);
        }
        
        for (int i = 0; i < K; i++)
        {
            MPI_Send(&objects[i].id, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Send(&objects[i].dim, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            int pixels = objects[i].dim * objects[i].dim;
            MPI_Send(objects[i].members, pixels, MPI_INT, 1, 0, MPI_COMM_WORLD);
        }
        
    } 
    else
    {
        for (int i = 0; i < numOfPics; i++)
        {
            MPI_Recv(&myPics[i].id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&myPics[i].dim, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            int pixels = myPics[i].dim * myPics[i].dim;
            myPics[i].members = (int*)calloc(pixels, sizeof(int));
            MPI_Recv(myPics[i].members, pixels, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        }

        objects = (Object*)calloc(K, sizeof(Object));
        for (int i = 0; i < K; i++)
        {
            MPI_Recv(&objects[i].id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&objects[i].dim, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            int pixels = objects[i].dim * objects[i].dim;
            objects[i].members = (int*)calloc(pixels, sizeof(int));
            MPI_Recv(objects[i].members, pixels, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        }
    }

    Findings* findings = searchObjectsInPictures(myPics, numOfPics, objects, K, matchingValue);

    if (rank == 0)
    {

        int slavePics = N - numOfPics;
        findings = (Findings*)realloc(findings, N * sizeof(Findings));
        if (!findings)
        {
            printf("Failure in memory reallocation\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        for (int i = 0; i < slavePics; i++)
            MPI_Recv(&findings[numOfPics + i], 10, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
        

        FILE* wf = fopen(OUTPUT, "w");
        if (!wf)
        {
            printf("Problem to open writing file\n");
            MPI_Abort(MPI_COMM_WORLD, 0);
        }
        for (int i = 0; i < N; i++)
        {
            if (findings[i].picId != EMPTY)
            {
                fprintf(wf, "Picture %d: found Objects: ", findings[i].picId);
                for (int k = 0; k < SIZE-1; k++)
                    fprintf(wf, "%d Position(%d,%d) ;", findings[i].objsId[k], findings[i].objsRows[k], findings[i].objsCols[k]);                
                        
                fprintf(wf, "%d Position(%d,%d)\n", 
                    findings[i].objsId[SIZE-1], findings[i].objsRows[SIZE-1], findings[i].objsCols[SIZE-1]);
            }   
            else
                fprintf(wf, "Picture %d: No three different Objects were found\n", pictures[i].id);   
        }
        
    } 
    else 
    {
        for (int i = 0; i < numOfPics; i++)
           MPI_Send(&findings[i], 10, MPI_INT, 0, 0, MPI_COMM_WORLD);    
    }

    if (rank == 0)
    {
        double end = MPI_Wtime();
        printf("Execution time: %.3fs\n", (end - start));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
