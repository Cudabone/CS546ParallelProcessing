#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
//Handle error function given
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
#define MPI_CHECK(fn) { int errcode; errcode = (fn); \
    if (errcode != MPI_SUCCESS) handle_error(errcode, #fn ); }

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

	//Set filename if given, else exit
	const char *filename;
	if(argc != 2)
	{
		if(rank == 0)
			printf("Usage: mpiexec -n <nprocs> ./rank <filename>\n");
		return -1;
	}
	else
	{
		filename = argv[1];
	}

	//Number of times to write rank
	int count = 10;
	//Set buffer size and Offset in file per write
	int bufsize = count*sizeof(int);
	MPI_Offset offset = rank*bufsize;

	//Create buffer filled with ranks
	int buf[count];
	int i;
	for(i = 0; i < count; i++)
		buf[i] = rank;

	//Open file
	MPI_CHECK(MPI_File_open(MPI_COMM_WORLD,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&file));
	//Write rank count times in file starting at offset
	MPI_CHECK(MPI_File_write_at(file,offset,buf,count,MPI_INT,NULL));
	
	//Close file and finish 
	MPI_CHECK(MPI_File_close(&file));
	MPI_Finalize();
	return 0;
}
