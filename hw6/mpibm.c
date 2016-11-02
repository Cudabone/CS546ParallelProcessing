#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MPI_CHECK(fn) { int errcode; errcode = (fn); \
    if (errcode != MPI_SUCCESS) handle_error(errcode, #fn ); }

static void handle_error(int errcode, char *str);
void store_output(const char *filename, int numMB,double wrtime, double rdtime, double ttime);


int rank, nprocs;

int main(int argc, char **argv)
{
	//Initialize MPI
	MPI_Init(&argc,&argv);
	//Set rank for each process
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	//Set size of the communicator
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_File file;

	//Timing variables
	double start,end;
	double wrstart,wrend;
	double rdstart,rdend;
	//start benchmark timer
	start = MPI_Wtime();

	//Define the size of a MegaByte
	const size_t MB = 1024*1024;

	//Set filename if given, else exit
	char *outfile = "/home/mmikuta/orangefs/storage/data/temp.dat";
	//Number of MB to write and read
	int numMB;
	//Output file for timings and bandwidth
	char *timefile;

	if(argc != 3)
	{
		if(rank == 0)
			printf("Usage: mpiexec -n <nprocs> ./mpibm <MB-per-rank> <filename2>\n");
		return -1;
	}
	else
	{
		//outfile = argv[1];
		numMB = atoi(argv[2])*MB;
		timefile = argv[3];
	}

	//Set buffer size and Offset in file per write
	MPI_Offset offset = rank*numMB;

	//Create buffer filled with ranks
	void *buf = malloc(numMB);
	/*
	int i;
	for(i = 0; i < count; i++)
		buf[i] = rank;
		*/

	/* FILE WRITE */
	//Start write timer
	wrstart = MPI_Wtime();
	//Open file
	MPI_CHECK(MPI_File_open(MPI_COMM_WORLD,outfile,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&file));
	//Write rank count times in file starting at offset
	MPI_CHECK(MPI_File_write_at(file,offset,buf,numMB,MPI_BYTE,NULL));
	MPI_CHECK(MPI_File_close(&file));
	//Stop write timer
	wrend = MPI_Wtime();

	/* FILE READ */
	//Start read timer
	rdstart = MPI_Wtime();
	//Open file
	MPI_CHECK(MPI_File_open(MPI_COMM_WORLD,outfile,MPI_MODE_RDONLY,MPI_INFO_NULL,&file));
	//Read
	MPI_CHECK(MPI_File_read_at(file,offset,buf,numMB,MPI_BYTE,NULL));
	MPI_CHECK(MPI_File_close(&file));
	//Stop read timer
	rdend = MPI_Wtime();

	free(buf);

	//Stop benchmark timer
	end = MPI_Wtime();
	double wrtime = (wrend-wrstart);
	double rdtime = (rdend-rdstart);
	double ttime = (end-start);

	//Write file
	if(rank == 0)
		store_output(timefile,numMB,wrtime,rdtime,ttime);
	

	MPI_Finalize();
	return 0;
}
//Store output timings
void store_output(const char *filename, int numMB, double wrtime, double rdtime, double ttime)
{
	FILE *fp = fopen(filename,"w");
	/*Store time in file*/
	fprintf(fp,"write time: %.2f seconds\n",wrtime);
	fprintf(fp,"read time: %.2f seconds\n",rdtime);
	fprintf(fp,"overall time: %.2f seconds\n",wrtime);
	double bwwrite = numMB/wrtime; 
	double bwread = numMB/rdtime;
	double maxbw = (bwwrite > bwread) ? bwwrite : bwread;
	fprintf(fp,"maximum bandwidth: %.2f MB/s\n",maxbw);
	
	/*Store each X value in file*/
	fclose(fp);
}
static void handle_error(int errcode, char *str)
{
	char msg[MPI_MAX_ERROR_STRING];
	int resultlen;
	MPI_Error_string(errcode, msg, &resultlen);
	fprintf(stderr, "%s: %s\n", str, msg);
	/* Aborting on error might be too aggressive.  If
	 *  * you're sure you can
	 *   * continue after an error, comment or remove
	 *    * the following line */
	MPI_Abort(MPI_COMM_WORLD, 1);
}
