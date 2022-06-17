#!/bin/bash
#SBATCH -J test18 #task name
#SBATCH -p normal #cluster partition
#SBATCH -N 50 #nodes number
#SBATCH --gres=dcu:4    #dcu number per node (max=4)
#SBATCH --ntasks-per-node=4 #cpu number per node(equal to processes per node)
#SBATCH --ntasks-per-socket=8 # task per socket
#SBATCH -o test18.o #output file
#SBATCH -e test18.e #error file
echo "Start time: `date`" #time
echo "SLURM_JOB_ID: $SLURM_JOB_ID" #task id
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"
echo '/public/home/ictapp_x/pyf_folder/comtest/svd_test18/bin/main'
mpirun  /public/home/ictapp_x/pyf_folder/comtest/svd_test18/bin/main # task
echo "End time: `date`"

