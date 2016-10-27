/* Matrix normalization.
 * Compile with "gcc matrixNorm.c" 
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

/* Program Parameters */
#define MAXN 8000  /* Max value of N */
int N;  /* Matrix size */

/* Matrices */
volatile float A[MAXN][MAXN], B[MAXN][MAXN];

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void matrixNorm();
void cudaErrorCheck(cudaError_t err, const char *s);
__global__ void matrixCuda(float *d_A,int N);
__global__ void BlockMean(float *d_A,float *d_Sum, int N);
__global__ void BlockDev(float *d_A, float *d_Dev, float *d_mu,int N);
__global__ void Normalize(float *d_A, float *d_mu, float *d_sigma, int N);

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

	if (argc == 3) {
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

/* Initialize A and B*/
void initialize_inputs() {
	int row, col;

	printf("\nInitializing...\n");

	for (col = 0; col < N; col++) {
		for (row = 0; row < N; row++) {
			A[row][col] = (float)rand() / 32768.0;
			B[row][col] = 0.0;
		}
	}
	/*
	   for (col = 0; col < N; col++) {
	   for (row = 0; row < N; row++) {
	   A[row][col] = col + row;
	   B[row][col] = 0.0;
	   }
	   }
	   */

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
	}
}

void print_B() {
	int row, col;

	if (N < 10) {
		printf("\nB =\n\t");
		for (row = 0; row < N; row++) {
			for (col = 0; col < N; col++) {
				printf("%1.10f%s", B[row][col], (col < N-1) ? ", " : ";\n\t");
			}
		}
	}
}

int main(int argc, char **argv) {
	/* Timing variables */
	struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
	struct timezone tzdummy;
	clock_t etstart2, etstop2;  /* Elapsed times using times() */
	unsigned long long usecstart, usecstop;
	struct tms cputstart, cputstop;  /* CPU times for my processes */

	/* Process program parameters */
	parameters(argc, argv);

	/* Initialize A and B */
	initialize_inputs();

	/* Print input matrices */
	print_inputs();

	/* Start Clock */
	printf("\nStarting clock.\n");
	gettimeofday(&etstart, &tzdummy);
	etstart2 = times(&cputstart);

	/* Gaussian Elimination */
	matrixNorm();

	/* Stop Clock */
	gettimeofday(&etstop, &tzdummy);
	etstop2 = times(&cputstop);
	printf("Stopped clock.\n");
	usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
	usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

	/* Display output */
	print_B();

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

	exit(0);
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][] and B[][],
 * defined in the beginning of this code.  B[][] is initialized to zeros.
 */


/* Matrix Normalization for the Cuda platform
   Overview
   1.Copy matrix A to Device
   2.CUDA: Split matrix into 2D grid of blocks, each block calculates a partial sum 
   for each column in their section via BlockMean.
   3.Sequentially add all of these partial sums to form the total mean for
   each column.
   4.CUDA: Each block calculate a partial sum of the mean difference for each
   column in their section via BlockDev.
   5.Sequentially add all of these partial mean differences to form the total
   standard deviation for each column.
   6.CUDA: Each block normalize their portion of the matrix using the means
   and standard deviations calculated.
   7.Copy matrix A to Host's B
   */

//Number of threads per block
#define BlockSize 32
void matrixNorm() 
{
	printf("Executing on GPU\n");

	//Set grid size to divide among number of threads
	int GridSize = ceil((float)N/BlockSize); 

	//Create CUDA grid and block size for matrix
	dim3 grid(GridSize,GridSize);
	dim3 block(BlockSize,BlockSize);

	//Device matrices and vectors for calculation 
	float *d_A; //the matrix 
	float *d_Sum; //a partial sum holder
	float *d_Dev; //a partial deviation holder
	float *d_mu; //a vector of means for each column
	float *d_sigma; //a vector of standard deviations for each column

	//Host copies of variables to initalize the device's copies to 0's
	float *h_Sum = (float *)malloc(GridSize*N*sizeof(float));
	float *h_Dev = (float *)malloc(GridSize*N*sizeof(float));
	float *h_mu = (float *)malloc(N*sizeof(float));
	float *h_sigma = (float *)malloc(N*sizeof(float));
	size_t size = N*N*sizeof(float);

	// Allocate Matrices on Devices 
	cudaErrorCheck(cudaMalloc((void **)&d_A,size), "cudaMalloc A");
	cudaErrorCheck(cudaMalloc(&d_Sum,GridSize*N*sizeof(float)), "cudaMalloc d_Sum" );
	cudaErrorCheck(cudaMalloc(&d_Dev,GridSize*N*sizeof(float)), "cudaMalloc d_Dev ");
	cudaErrorCheck(cudaMalloc((void **)&d_mu,N*sizeof(float)), "cudaMalloc d_mu");
	cudaErrorCheck(cudaMalloc((void **)&d_sigma,N*sizeof(float)), "cudaMalloc d_sigma");

	// Copy over matrix to device 
	cudaErrorCheck(cudaMemcpy(d_A,(const void
					*)A[0],size,cudaMemcpyHostToDevice), "cudaMemcpy A");

	//Initalize h_sum and h_dev to 0
	int row,col;
	for(row = 0; row < GridSize; row++)
	{
		h_mu[row] = 0.0;
		h_sigma[row] = 0.0;
		for(col = 0; col < N; col++)
		{
			h_Sum[row*N + col] = 0.0;
			h_Dev[row*N + col] = 0.0;
		}
	}

	//Initalize the device sum and std dev arrays to 0s.
	cudaErrorCheck(cudaMemcpy((void *)d_Sum,(const void
				*)h_Sum,GridSize*N*sizeof(float),cudaMemcpyHostToDevice),
			"cudaMemcpy to d_Sum");
	cudaErrorCheck(cudaMemcpy((void *)d_Dev,(const void
				*)h_Dev,GridSize*N*sizeof(float),cudaMemcpyHostToDevice),"cudaMemcpy to d_Dev" );
	cudaErrorCheck(cudaMemcpy((void *)d_mu,(const void
				*)h_mu,N*sizeof(float),cudaMemcpyHostToDevice), "cudaMemcpy to d_mu" );
	cudaErrorCheck(cudaMemcpy((void *)d_sigma,(const void
				*)h_sigma,N*sizeof(float),cudaMemcpyHostToDevice), "cudaMemcpy to d_sigma");


	//Calcuate a sub mean for each block
	BlockMean<<<grid,block>>>(d_A,d_Sum,N);
	cudaDeviceSynchronize();

	//Calculate total mean for each column sequentially 
	cudaErrorCheck(cudaMemcpy((void *)h_Sum,(const void
					*)d_Sum,GridSize*N*sizeof(float),cudaMemcpyDeviceToHost),
			"cudaMemcpy to h_Sum");

	for(row = 0; row < GridSize; row++)
	{
		for(col = 0; col < N; col++)
		{
			h_mu[col] += h_Sum[row*N + col];
		}
	} 
	for(col = 0; col < N; col++)
		h_mu[col] /= N; 

	//Copy over host calculated mu vector to host
	cudaErrorCheck(cudaMemcpy((void *)d_mu,(const void
				*)h_mu,N*sizeof(float),cudaMemcpyHostToDevice), "cudaMemcpy to d_mu");

	//Calculate a sub standard deviation for each block
	BlockDev<<<grid,block>>>(d_A,d_Dev,d_mu,N);
	cudaDeviceSynchronize();

	//Calculate total standard deviation from each block sequentially
	cudaErrorCheck(cudaMemcpy((void *)h_Dev,(const void
				*)d_Dev,GridSize*N*sizeof(float),cudaMemcpyDeviceToHost),
			"cudaMemcpy to h_dev");

	for(row = 0; row < GridSize; row++)
	{
		for(col = 0; col < N; col++)
		{
			h_sigma[col] += h_Dev[row*N + col];
		}
	}
	for(col = 0; col < N; col++)
	{
		h_sigma[col] /= N;
		h_sigma[col] = sqrt(h_sigma[col]);
	}

	//Copy sigma vector to device
	cudaErrorCheck(cudaMemcpy((void *)d_sigma,(const void
				*)h_sigma,N*sizeof(float),cudaMemcpyHostToDevice), "cudaMemcpy to d_sigma" );

	//Normalize with means and standard deviations by splitting into blocks
	Normalize<<<grid,block>>>(d_A,d_mu,d_sigma,N);
	cudaDeviceSynchronize();

	//Copy Normalized array back to B
	cudaErrorCheck(cudaMemcpy((void *)B[0],(const void
					*)d_A,size,cudaMemcpyDeviceToHost), "cudaMemcpy to B");

	//Free all host and device pointers
	cudaFree(d_A);
	cudaFree(d_Sum);
	cudaFree(d_Dev);
	cudaFree(d_mu);
	cudaFree(d_sigma);
	free(h_Sum);
	free(h_Dev);
	free(h_mu);
	free(h_sigma);
}

/* Calculates a partial sum for a section of the matrix for each block.
   This is done by allocating a shared sub matrix, calculated a
   sum for each column using the algorithm from class, and returning
   the corresponding sub sum from each block.
   d_Sum holds all of these partial sums for every block.
   */
__global__ void BlockMean(float *d_A,float *d_Sum, int N)
{
	//Shared sub matrix
	__shared__ float sum[BlockSize*BlockSize];

	//Block and thread indices
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	//Size of entire matrix
	int size = N*N*sizeof(float);

	//Indices for flattened input matrices
	// i + j == [i][j]
	unsigned int i = blockIdx.x*BlockSize*N + tx*N;
	unsigned int j = blockIdx.y*BlockSize + ty;

	//Row index into sum
	unsigned int sx = tx*BlockSize;

	//Ensure block and thread within bounds of matrix
	if(x >= N || y >= N)
		return; 

	//Transfer section of d_A into sum
	if(i + j < size) 
		sum[sx + ty] = d_A[i + j];  
	else 
		sum[sx + ty] = 0.0;

	//Apply partial sum algorithm from class
	for(unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
	{
		__syncthreads();
		if(tx < stride)
			sum[sx + ty] += sum[sx + ty + stride];
	}
	
	//Transfer shared sub sum matrix to global memory
	if(tx == 0)
	{
		d_Sum[blockIdx.x*N + ty] = sum[ty];
	}
}

/* Calculate a partial sum of the square of the difference between the mean 
   for each block, in the same way as BlockMean, except by squaring a 
   difference of a calculated mean for the column in d_mu.
   */
__global__ void BlockDev(float *d_A, float *d_Dev, float *d_mu, int N)
{
	//Shared mu vector
	__shared__ float mu[BlockSize];
	//shared partial sum sub matrix
	__shared__ float sum[BlockSize*BlockSize];
	//Size of entire matrix
	int size = N*N*sizeof(float);

	//Block and thread indices
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	//Indices for flattened input matrices
	// i + j == [i][j]
	unsigned int i = blockIdx.x*BlockSize*N + tx*N;
	unsigned int j = blockIdx.y*BlockSize + ty;

	//Row index into sum
	unsigned int sx = tx*BlockSize;

	//Ensure block and thread within bounds of matrix
	if(x >= N || y >= N)
		return; 

	//Transfer sub mu vector into shared memory
	if(tx == 0)
	{
		mu[ty] = d_mu[j];
	}
	
	//Ensure mu vector populated
	__syncthreads();

	//Transfer section of d_A into sum and square the mean difference
	if((i + j < size) && (j < N))
		sum[sx + ty] = powf(d_A[i + j] - mu[ty],2.0);  
	else 
		sum[sx + ty] = 0.0;

	
	//Apply partial sum algorithm shown in class
	for(unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
	{
		__syncthreads();
		if(tx < stride)
			sum[sx + ty] += sum[sx + ty + stride];
	}

	//Tranfer shared sub matrix to global memory
	if(tx == 0)
	{
		d_Dev[blockIdx.x*N + ty] = sum[ty];
	}
}
/* Normalizing function, which each thread among all blocks corresponds
   to a single element in the matrix. Each applies the normalizing function
   to its element*/
__global__ void Normalize(float *d_A, float *d_mu, float *d_sigma, int N)
{
	// Thread position variables
	unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;

	//Ensure within bounds of matrix
	if(tx >= N || ty >= N)
		return;

	//If sigma 0, set 0, else calculate normalized value
	if(d_sigma[ty] == 0)
		d_A[tx + ty*N] = 0;
	else
		d_A[tx + ty*N] = (d_A[tx + ty*N] - d_mu[ty])/(d_sigma[ty]);
}

/*Simple printing error function */
void cudaErrorCheck(cudaError_t err, const char *s)
{
	if(err != cudaSuccess)
	{
		printf("%s error: %s\n",s,cudaGetErrorString(err));
		exit(0);
	}
}

