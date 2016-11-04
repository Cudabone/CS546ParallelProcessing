#!/bin/bash
#Arg 1: Procs per node
#Arg 2: MB to read/write
#Arg 3: Output file
qsub -pe mpich 4 runbm.sh $1 $2
