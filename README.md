# What's this
this is a batched SVD implementation (names W-Cycle_SVD) test on NVIDIA CUDA platform

# Hardware env:
- NVIDIA GPU with tensor core supported (we used a100)

# Software env
- Linux system(we used CentOS 7.9.2009)
- CUDA tookit (we used 11.6)
- g++ (we used 4.8.5)

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
