# What's this
this is a batched SVD implementation (names W-Cycle_SVD) test on NVIDIA CUDA platform

# Hardware env:
- NVIDIA GPU (we used Tesla V100)

# Software env
- Linux system(we used Ubuntu18.04)
- CUDA tookit (we used 10.1)
- g++ (we used 7.5)

# how to built this program:
run these command in the main directory
```shell
make
````
Then you will get a executable file names 'test' in the main directory.
Before run it, you should prepare necessary data first
```shell
cd data
unzip UF_matrixset.zip
cd ..
./test 99 # generate some size-specified random matrices
```
After run these commands,
you will see 5 folders under the "data/UF_matrixset" directory, matrices in these floders are necessary in test 10
And there will be some matrices In the "data/generated_matrixes" folder too, they are named by their shape.
 
then run `./test ${TSETINDEX}`, with `${TSETINDEX}` recommand the index of all the 10 test.
Like this：
```
./test 1
# this command will run test1
```
