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

#include <mpi.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N */
int N;	/* Matrix size */
/* globals rank and number of processes */
int rank, nprocs;

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
		if(rank ==0)
			printf("Random seed = %i\n", seed);
	} 
	if (argc >= 2) {
		N = atoi(argv[1]);
		if (N < 1 || N > MAXN) {
			if(rank == 0)
			printf("N = %i is out of range.\n", N);
			exit(0);
		}
	}
	else {
		if(rank ==0)
			printf("Usage: %s <matrix_dimension> [random seed]\n",
					 argv[0]);		
		exit(0);
	}

	/* Print parameters */
	if(rank == 0)
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
	//int rank,nprocs; //Rank and number of processes for MPI
	double start, end; // Timer for parallel execution
	/* Initialize parallel environment, number of processes and their ranks */
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	/* Timing variables */
	/*
	struct timeval etstart, etstop;  // Elapsed times using gettimeofday() //
	struct timezone tzdummy;
	clock_t etstart2, etstop2;	// Elapsed times using times() //
	unsigned long long usecstart, usecstop;
	struct tms cputstart, cputstop;  // CPU times for my processes //
	*/

	/* Process program parameters */
	parameters(argc, argv);
	/*Wait for all processes to parse parameters*/
	MPI_Barrier(MPI_COMM_WORLD);
	/* Initialize A and B */
	if(rank == 0)
	{
		initialize_inputs();
	}
	int row,col;
	/* Send initialized inputs to all processes: A,B,and X */
	for(row = 0; row < N; row++)
	{
		if(rank == 0)
		{
			MPI_Bcast(A[row],N,MPI_FLOAT,0,MPI_COMM_WORLD);
		}
		else if(rank !=0)
			MPI_Bcast(A[row],N,MPI_FLOAT,0,MPI_COMM_WORLD);
	}
	MPI_Bcast(&B[0],N,MPI_FLOAT,0,MPI_COMM_WORLD);
	if(rank !=0){
	for(col = 0; col < N; col++)
		X[col] = 0.0;
	}

	/* Print input matrices */
	if(rank == 0)
		print_inputs();

	/* Start Clock */
	/*if(rank ==0){
	printf("\nStarting clock.\n");
	gettimeofday(&etstart, &tzdummy);
	etstart2 = times(&cputstart);
	}*/
	MPI_Barrier(MPI_COMM_WORLD);
	//Start clock
	if(rank ==0)
		printf("Starting clock\n");
	start = MPI_Wtime();
	/* Gaussian Elimination */
	gauss();

	//if(rank == 0){

	/* Stop Clock */
		/*
	gettimeofday(&etstop, &tzdummy);
	etstop2 = times(&cputstop);
	printf("Stopped clock.\n");
	usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
	usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

	// Display output 
	print_X();
	// Display timing results 
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
			// Contrary to the man pages, this appears not to include the parent //
	printf("--------------------------------------------\n");
	}
	*/
	if(rank == 0)
	{
		//Stop MPI Clock
		end = MPI_Wtime();
		printf("Stopping clock\n");
		//Print output
		print_X();
		printf("Parallel execution time: %.3f ms\n",(end-start)*1000);
		print_inputs();
		//Store output in file
		if(argc == 4)	
			store_output(argv[3]);
		else
			printf("Usage: %s <matrix_dimension> [random seed] <filename>, for an output file\n",argv[0]);
	}
	
	MPI_Finalize();
	
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
void gauss() {
	/*if(rank !=0)
		int temp[N];
	else if(rank == 0)
	{
		int temp
	}*/

	if(rank ==0)
		printf("Computing in Parallel.\n");


	/* Gaussian elimination */
	/* For each row iteration, you modify the entire row */
	/* Each iteration of norm loop restarts the row modifications
	 * Each NORM: Modify Row norm+1 -> end, Row norm+2 -> end, Row norm+3 -> end 
	 * as well as B matrix */
	/* Dependency on each iteration of norm for each row to be calculated */

	int norm; //Normalization row and zeroing
	int row, col; /* element row and col */
	for (norm = 0; norm < N - 1; norm++) {
		for (row = norm + 1; row < N; row++) {
			float multiplier;
			/* Divide row calculations among the number of processes */
			if(row % nprocs == rank)
			{
				multiplier = A[row][norm] / A[norm][norm];
				for (col = norm; col < N; col++) {
					A[row][col] -= A[norm][col] * multiplier;
				}
				B[row] -= B[norm] * multiplier;
			}
		}
		/* If normalization is not complete, the row A[norm] will be needed
		 * the next iteration, so broadcast the row from the process who
		 * calculated it 
		 * The broadcast statement also acts as a barrier before the next 
		 * iteration of normalization which requires that all processes
		 * completed their work from the previous iteration to remain correct.*/
		if(norm+1 < N-1) 
		{
			MPI_Bcast(A[norm+1],N,MPI_FLOAT,(norm+1) % nprocs,MPI_COMM_WORLD);
			MPI_Bcast(&B[norm+1],1,MPI_FLOAT,(norm+1) % nprocs,MPI_COMM_WORLD);
		}
	}
	/*Before back substitution all of the calculations over B and the last row
	 * of A need to be sent to a single process, in this case rank 0. Only
	 * sending the last row of A minimizes communication.*/
	for(row = 0; row < N; row++)
	{
		/*Send each calculated B row to rank 0 from corresponding process who calculated 
		 * it*/
		if((row % nprocs == rank) && (rank !=0))
			MPI_Send(&B[row],1,MPI_FLOAT,0,0,MPI_COMM_WORLD);
		else if((row % nprocs != 0) && (rank == 0))
			MPI_Recv(&B[row],1,MPI_FLOAT,(row % nprocs),0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	}
	if((N-1) % nprocs == rank)
		MPI_Send(A[N-1],N,MPI_FLOAT,0,0,MPI_COMM_WORLD);
	else if(rank == 0)
		MPI_Recv(A[N-1],N,MPI_FLOAT,((N-1) % nprocs),0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	//MPI_Barrier(MPI_COMM_WORLD); //Ensure all processes done before back substitution
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

