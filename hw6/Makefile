CC = mpicc
CFLAGS = -Wall
INCLUDES = -I /home/mmikuta/orangefs/include #-I /mpich2_install_dir/include 
LIBS = -L /home/mmikuta/orangefs/lib #-L /mpich2_install_dir/lib 
MPIFLAGS = -lpvfs2 

rank: rank.c
	$(CC) $(CFLAGS) rank.c -o rank

bm: mpibm.c
	$(CC) $(CFLAGS) mpibm.c -o mpibm

bmjarvis: mpibm.c
	$(CC) $(CFLAGS) mpibm.c -o mpibm $(INCLUDES) $(LIBS) $(MPIFLAGS) 

.PHONY: clean

clean:
	@rm -f rank *.dat mpibm
#mpibm.c -o mpibm -L /home/mmikuta/orangefs/lib -L /mpich2_install_dir/lib -I /home/mmikuta/orangefs/include -I /mpich2_install_dir/include -lpvfs2 -lmpich
