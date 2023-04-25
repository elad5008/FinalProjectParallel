build:
	mpicxx -c main.c -o main.o
	mpicxx -fopenmp -c cFunctions.c -o cFunctions.o
	nvcc -I./inc -gencode arch=compute_61,code=sm_61 -c CudaFunctions.cu -o CudaFunctions.o
	mpicxx -fopenmp -o runFile main.o cFunctions.o CudaFunctions.o /usr/local/cuda/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./runFile
	rm -f output.txt Serial

run:
	mpiexec -np 2 -hostfile hosts ./runFile
