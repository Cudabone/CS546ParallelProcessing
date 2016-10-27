CC = gcc
CFLAGS = -Wall
FILES = matrixNorm.c
MAIN: ./matrixNorm

all: $(MAIN)

cuda: matrixNorm.cu
	nvcc matrixNorm.cu -o matrixCuda

clean:
	@rm -f ./matrixNorm ./matrixCuda
