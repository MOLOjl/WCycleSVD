# What's this
this is a batched SVD implementation (names W-Cycle_SVD) test on NVIDIA CUDA platform

# Hardware env:
- AMD GPU (we used vega20)

# Software env
- Linux system(we used Ubuntu18.04)
- ROCm tookit (we used 3.3)
- g++ (we used 7.5)
- magma (based on HIP and mkl)

# how to built this program:
Check and change the dependency directories in 'Makefile'.
Run 'make' command in the main directory
```shell
make
````
Then you will get a executable file names 'svd' in the main directory.
Just use './' to run it.
```shell
./svd
```
