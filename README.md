# What's this
this is a batched SVD implementation (names W-Cycle_SVD) test on NVIDIA CUDA platform

# Hardware env:
- NVIDIA GPU with tensor core supported (we used a100)

# Software env
- Linux system(we used Ubuntu18.04)
- CUDA tookit (we used 10.1)
- g++ (we used 7.5)

# how to built this program:
run 'make' command in the main directory
```shell
make
````
Then you will get a executable file names 'test' in the main directory.
Run:
```shell
./test 16
```
