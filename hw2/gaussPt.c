/* Gaussian elimination without pivoting.
 * Compile with "gcc gauss.c" 
 */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>

#include <pthread.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N */
#define MAXTHREADS 100 /* max number of threads */
int N;	/* Matrix size */
/* Number of Pthreads to run */
int nprocs = 4;
/* Pthread Barrier */
pthread_barrier_t *barrier;

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
		* It is this routine that is timed.
		* It is called only on the parent.
		*/
/*Pthread helper function for gaussian elimination */
void gaussPt();

/* Store output prototype to save output to file*/
void store_output(char *);

/* returns a seed for srand based on the time */
unsigned int time_seed() {
	struct timeval t;
	struct timezone tzdummy;

	gettimeofday(&t, &tzdummy);
	return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
	int seed = 0;  /* Random seed */
	char uid[32]; /*User name */

	/* Read command-line arguments */
	srand(time_seed());  /* Randomize */

	if (argc >= 3) {
		seed = atoi(argv[2]);
		srand(seed);
		printf("Random seed = %i\n", seed);
	} 
	if (argc >= 2) {
		N = atoi(argv[1]);
		if (N < 1 || N > MAXN) {
			printf("N = %i is out of range.\n", N);
			exit(0);
		}
	}
	else {
		printf("Usage: %s <matrix_dimension> [random seed]\n",
					 argv[0]);		
		exit(0);
	}

	/* Print parameters */
	printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
	int row, col;

	printf("\nInitializing...\n");
	for (col = 0; col < N; col++) {
		for (row = 0; row < N; row++) {
			A[row][col] = (float)rand() / 32768.0;
		}
		B[col] = (float)rand() / 32768.0;
		X[col] = 0.0;
	}

}

/* Print input matrices */
void print_inputs() {
	int row, col;

	if (N < 10) {
		printf("\nA =\n\t");
		for (row = 0; row < N; row++) {
			for (col = 0; col < N; col++) {
	printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
			}
		}
		printf("\nB = [");
		for (col = 0; col < N; col++) {
			printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
		}
	}
}

void print_X() {
	int row;

	if (N < 100) {
		printf("\nX = [");
		for (row = 0; row < N; row++) {
			printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
		}
	}
}

int main(int argc, char **argv) {
	/* Timing variables */
	struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
	struct timezone tzdummy;
	clock_t etstart2, etstop2;	/* Elapsed times using times() */
	unsigned long long usecstart, usecstop;
	struct tms cputstart, cputstop;  /* CPU times for my processes */

	/* Process program parameters */
	parameters(argc, argv);

	/* Initialize A and B */
	initialize_inputs();

	/* Print input matrices */
	print_inputs();


	/*Number of threads to run*/

	/*Create threads*/
	/* Create space for pointers to threads */
	/*
	if(argc != 2)
	{
		printf("Usage: %s n, where n is the number of threads\n",argv[0]);
		exit(1);
	}
	int nthreads = atoi(argv[1]);
	if(n < 1 || n > MAXTHREADS)
	{
		printf("Number of threads must be between 1 and %d\n",MAXTREADS);
		exit(1);
	}
	*/

	/* Start Clock */
	printf("\nStarting clock.\n");
	gettimeofday(&etstart, &tzdummy);
	etstart2 = times(&cputstart);

	printf("Computing in parallel with %d threads",nprocs);

	/* Gaussian Elimination */
	gaussPt();

	/* Stop Clock */
	gettimeofday(&etstop, &tzdummy);
	etstop2 = times(&cputstop);
	printf("Stopped clock.\n");
	usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
	usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

	/* Display output */
	print_X();

	/* Display timing results */
	printf("\nElapsed time = %g ms.\n",
	 (float)(usecstop - usecstart)/(float)1000);

	printf("(CPU times are accurate to the nearest %g ms)\n",
	 1.0/(float)CLOCKS_PER_SEC * 1000.0);
	printf("My total CPU time for parent = %g ms.\n",
	 (float)( (cputstop.tms_utime + cputstop.tms_stime) -
			(cputstart.tms_utime + cputstart.tms_stime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
	printf("My system CPU time for parent = %g ms.\n",
	 (float)(cputstop.tms_stime - cputstart.tms_stime) /
	 (float)CLOCKS_PER_SEC * 1000);
	printf("My total CPU time for child processes = %g ms.\n",
	 (float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
			(cputstart.tms_cutime + cputstart.tms_cstime) ) /
	 (float)CLOCKS_PER_SEC * 1000);
			/* Contrary to the man pages, this appears not to include the parent */
	printf("--------------------------------------------\n");

	print_inputs();
	/* Store X output to file */
	if(argc == 4)	
		store_output(argv[3]);
	else
		printf("Usage: %s <matrix_dimension> [random seed] <filename>, for an output file\n",argv[0]);
	
	exit(0);
}
void store_output(char *filename)
{
	int row;
	FILE *fp = fopen(filename,"w");
	for(row = 0; row < N; row++)
	{
		if(fprintf(fp,"%.2f\n",X[row])< 0)
			perror("Output file Write error");
		/*if(fwrite(&X[row],sizeof(float),1,fp)< 0);
		if(fwrite())
		*/
	}
	fclose(fp);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */
void gauss(void *r) {
	int rank = (int)r; 
	/* Gaussian elimination */
	/* For each row iteration, you modify the entire row */
	/* Each iteration of norm loop restarts the row modifications
	 * which is a dependency and must be barriered */
	/* Each NORM: Modify Row norm+1 -> end, Row norm+2 -> end, Row norm+3 -> end 
	 * as well as B matrix */
	/* Dependency on each iteration of norm for each row to be calculated */

	int norm; //Normalization row and zeroing
	int row, col; //element row and col //
	for (norm = 0; norm < N - 1; norm++) {
		for (row = norm + 1; row < N; row++) {
			float multiplier;
			if(row % nprocs == rank)
			{
				multiplier = A[row][norm] / A[norm][norm];
				for (col = norm; col < N; col++) {
					A[row][col] -= A[norm][col] * multiplier;
				}
				B[row] -= B[norm] * multiplier;
			}
		}
		/*Barrier, dependency on completion of all rows for each norm
		iteration, specifically modifications to matrices A and B. */
		pthread_barrier_wait(barrier);
	}
	pthread_barrier_wait(barrier); //Ensure all processes done before back substitution
	/* (Diagonal elements are not normalized to 1.	This is treated in back
	 * substitution.)
	 */

	if(rank == 0)
	{
	/* Back substitution */
	for (row = N - 1; row >= 0; row--) {
		X[row] = B[row];
		for (col = N-1; col > row; col--) {
			X[row] -= A[row][col] * X[col];
		}
		X[row] /= A[row][row];
	}
	}
}
/* Helper function for the pthreads implementation of gaussian elimination.
 * This function creates the needed threads and calls gauss() in which each
 * thread will calculate their share of the matrix. The threads are then 
 * terminated.*/
void gaussPt()
{
	//Create space for n threads
	pthread_t threads[nprocs];
	//Allocate and create a barrier that blocks for all threads
	barrier = malloc(sizeof(pthread_barrier_t));
	pthread_barrier_init(barrier,NULL,nprocs);
	int i;
	//Create the n threads
	for(i = 0; i < nprocs; i++)
	{
		/* Call gauss with each thread and paremeter i which
		 * will act like a rank in MPI */
		pthread_create(&threads[i],NULL,gauss,(void *)i);
	}
	/* Terminate all threads */
	for(i = 0; i < nprocs; i++)
	{
		pthread_join(threads[i],NULL);
	}
}

