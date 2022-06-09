# What's this
this is a batched SVD implementation (names W-Cycle_SVD) test on NVIDIA CUDA platform

# Hardware env:
- NVIDIA GPU (we used Tesla V100)

# Software env
- Linux system with basic software environment(we used Ubuntu18.04)
- CUDA tookit (we used 10.1)
- g++ (we used 7.5)
- [magma](https://icl.utk.edu/magma/software/index.html) (based on cuda and [onemkl](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html), we used 2.5.4)

# How to built and run this project:
Run this command in the main directory
```shell
make
````
Then you will get a executable file names 'test' in the main directory. and a executable file names 'ma' in the 'magma_test' directory,
Before run 'test', you should prepare necessary data first
```shell
cd data
unzip UF_matrixset.zip
cd ..
./test 99 # generate some size-specified random matrices
```
After run these commands,
you will see 5 folders under the "data/UF_matrixset" directory, matrices in these floders are necessary in test 15.
And there will be some matrices In the "data/generated_matrixes" folder too, they are named by their shape.
 
Then run `./test ${TSETINDEX}`, with `${TSETINDEX}` recommand the index of the tests.
Like this
```
./test 1
# this command will run test1
```
Those tests are corresponded with the experiments in our paper, you can see the corresponding relationship from the annotations of `src/test.cu`
