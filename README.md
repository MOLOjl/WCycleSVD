# what's this 
this is a batched SVD implementation test on NVIDIA CUDA platform
# how to built this program:

run these command
```shell
make
````
then you will get a executable file names 'test' in the main directory
before run it,you should prepare data first
```shell
cd data
unzip UF_matrixset.zip
```
you will see 5 folder under the "data/" directory, matrix in these floder are necessary in test 5
```shell
./test 99
```
this step will generate some matrixes neeeded in test1-4, its choosable, but recommended

test 1-5 can be run by add the index after the "./test" command, like this
```
./test 1
# this command will run test1
```
test 6 is a user specified SVD test, you need to rewrite the `io_files/in.txt` file to specify the matrix shape and the tailoring strategy(tile shape), then you can just run
```
./test
```

it will load matrix from a certain file(both name and data storage are in regulation form), the imply SVD to it, as to validate the result's correctness, we use cuSOLVER' SVD as a benchmark, the singular value from our SVD and cuSOLVER will be output into two `out.txt` and `out_lib.txt` files in `io_files/`

for exsample, if `in.txt` is writed like this
```
10 512 512 256 32
```
then `./test` command will read `io_file/A_h512_w512.txt` to load a 512 × 512 matrix, copy 10 times, and imply SVD to these 10 matrixes. the tile size during SVD is set as 256 × 32 
