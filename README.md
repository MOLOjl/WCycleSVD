W-cycle SVD is a multilevel algorithm for batched SVD on GPUs. W-cycle SVD is size-oblivious, which successfully exploits the data reuse and ensures the optimal convergence speed for multiple SVDs. To push the envelope of performance further, we design the efficient batched SVD and EVD kernels, and propose a tailoring strategy to accelerate batched GEMMs in SVDs. This repository includes the full source code of W-cycle SVD program and some appended contents for the convenience of those experiments reported in our paper.

# Abstract Checklist
## Platforms:
### NVIDIA CUDA platform
#### The GPUs we used are
- Tesla V100 
- Tesla P100 
- GTX Titan X
        
### AMD ROCm platform 
#### The GPU we used are: 
- Vega20 

## System Details:
- 18.04-Ubuntu x86\_64 GNU/Linux (V100, P100 and TiTan X)
- CentOS 7.9 (A100)
- CentOS 7.6 (AMD GPU)

## Software Dependencies:
- GNU Make 4.1 
- CUDA toolkit (tested 10.1, 11.6) 
- nvprof 
- gcc/g++ (tested 4.8.5, 7.5) 
- ROCm (tested 3.5, 4.2) 
- Intel oneMKL (tested 2022.1.0) 
- MAGMA (tested 2.5.4)


# Environment Setup
## Basic environment
### CUDA Platform:
CUDA toolkit (version more than 10.1) should be installed. The compiler used is nvcc. Extra libraries needed are MAGMA, cuSOLVER and cuBLAS. The MAGMA we used depends on CUDA toolkit and intel oneMKL.

### ROCm Platform (single GPU):
ROCm toolkit (version more than 4.2) should be installed. The compiler used is hipcc. Extra libraries needed are MAGMA(hip), The MAGMA we used depends on ROCm toolkit and intel oneMKL.

## Compile the program
The project can be accessed on the Github by this [link](https://github.com/MOLOjl/WCycleSVD).

Use `git` (`http`, `ssh`, etc.) to clone the repository into a local directory.

For the 4 environments on which our artifact was mainly tested, there are 4 branches:
- main\_CUDA,
- test\_Tensor\_Core,
- test\_HIP,
- test\_Cluster

After cloned the branch corresponding to the environment. Run `make` in the root directory.

## Prepare necessary data
For the `main_CUDA` branch, The data are too large to store in the repository of Github. Please generate them by running the following command:
```shell
unzip data/UF_matrixset.zip
./test 99
```

# Experiments list
This list shows all the experiments in the [revision paper](https://submissions.supercomputing.org/?args=Aprcnt3DxGQUIYprcnt3Db0bf3UIIQs0f6IbQzTJUHtGrUAr0bf3UIIQs0bfPCzfx0rxJTxprcnt3D0Hprcnt3DI0rbCHIGdbUfTzCxGrQr_chz9Tz0Cx0zfsGRcMa9THQP0Aprcnt3DxfGzU3ACIIfb0HQP0Aprcnt3DxfTtUbprcnt3DsfGQUIYprcnt3DbTtUbb0XfQbGzt99TzYprcnt3D40bprcnt3DQxGdbUfTzYprcnt3D40QHHGdbUfTzYprcnt3D40Iprcnt3Dxprcnt3DGdbUfTrAprcnt3DxGPCf40zU3b0bfPzTrJUHtGPCf40zU3b0bfPzTEGpMcMN).

## V100, P100 and GTX TiTan X (`main_CUDA branch`):

Time of one-sided Jacobi methods in different cases. (Fig.1) <br>
Run: `./test 1` <br>
One-sided Jacobi method for a batched SVD of 100 matrices with each size of 1536×1536. (Fig 2) <br>
Run: `./test 2` <br>
Different tile sizes for two batched GEMMs at Level 1 of W-cycle SVD with two levels for 100 matrices. (TABLE I) <br>
Run: `./test 3` <br>
W-cycle SVD for improvement over cuSOLVER with matrix size below 32. (Fig.7) <br>
Run: `./test 4` <br>
Comparison with cuSOLVER using batch size=1 with matrix size between 500 and 10000. (Fig.8(a)) <br>
Run: `./test 5` <br>
W-cycle SVD for performance improvement with matrix size between 64 and 1024. (Fig.8(b))<br>
Run: `./test 6` <br>
W-cycle SVD for improvement over MAGMA. (Fig.9) <br>
Run: `./test 7` <br>
Time(s) for SVDs of 200 Matrices on P100 GPU. (TABLE IV) <br>
Run: `./test 8` <br>
Evaluation on different approaches in W-cycle SVD, one warp or $\alpha$ warps (Fig.10(a)) <br>
Run: `./test 9` <br>
Evaluation on different approaches in W-cycle SVD, original or parallel EVD. (Fig 10(b)) <br>
Run: `./test 10` <br>
GPU occupancy. (Fig.11(a)) <br>
Run: `./test11.sh` <br>
GM transaction. (Fig.11(b)) <br>
Run: `./test12.sh` <br>
Improvements of the tailoring strategy. (Fig.12) <br>
Run: `./test 13` <br>
Time(s) of W-cycle SVD with different tailoring plans. (TABLE V) <br>
Run: `./test 14` <br>
Evaluation of W-cycle SVD with various matrix sizes, with SuiteSparse matrix set. (TABLE VI) <br>
Run: `./test 15` <br>
Sensitivity on different GPUs. (Fig 14(a)) <br>
Run: `./test 17` <br>
Evaluation on the accuracy and convergence speed. (TABLE VII) <br>
Run: `./test 18` <br>
Evaluation on the accuracy. (Fig 15(a)) <br>
Run: `./test 19` <br>
Evaluation on convergence speed. (Fig 15(b)) <br>
Run: `./test 20` <br>

## A100 (test_Tensor_Core branch):
Evaluation on A100 GPU with tensor cores (Fig.13) <br>
Run: `./test 16` <br>

## Vega20 (test\_HIP branch):
Sensitivity on different GPUs. (Fig 14(a)) <br>
Run: `./svd` <br>

## GPU cluster (test_Cluster branch):
Data assimilation application. (Fig 14(b)) <br>
Run: `sbatch test18.slurm` <br>
The number of GPUs used is defined in the `test18.slurm` script. After the program finished, the result will write in `test18.o`.<br>
