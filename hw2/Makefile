CFLAGS = -Wall

MPI:
	mpicc $(CFLAGS) gaussMPI.c -o gaussMPI

seq:
	gcc $(CFLAGS) gauss.c -o gauss

Pt:
	gcc $(CFLAGS) -pthread gaussPt.c -o gaussPt

clean:
	rm ./gaussMPI ./gaussPt ./gauss 
