#!/bin/bash
#SBATCH -J test18 #指定作业名称
#SBATCH -p normal #指定分区名称
#SBATCH -N 2 #指定节点数量
#SBATCH --gres=dcu:4    #dcu数量
#SBATCH --ntasks-per-node=4 #指定每节点的任务数量，默认每个任务一个 CPU 核心
#SBATCH --ntasks-per-socket=8 # 任务绑定
#SBATCH -o test18.o #指定正确输出文件名称
#SBATCH -e test18.e #指定报错信息输出文件名称
echo "Start time: `date`" #显示开始时间
echo "SLURM_JOB_ID: $SLURM_JOB_ID" #显示作业号
echo "SLURM_NNODES: $SLURM_NNODES" #显示节点数
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE" #显示每节点任务数
echo "SLURM_NTASKS: $SLURM_NTASKS" #显示总任务数
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION" #显示作业分区
echo '/public/home/ictapp_x/pyf_folder/comtest/svd_test18/bin/main' #显示执行程序，用户可以根据自己程序进行修改
mpirun  /public/home/ictapp_x/pyf_folder/comtest/svd_test18/bin/main #此处用户需要换成自己的 MPI 程序
echo "End time: `date`" #显示结束时间

