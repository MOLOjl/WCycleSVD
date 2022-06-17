# What's this
this is a batched SVD implementation (names W-Cycle_SVD) test on a ROCm GPU cluster platform

# Hardware env:
- ROCm GPU (we used Vega20)

# Software env
- Linux system
- MPI
- ROCm tookit
- MAGMA-hip
- g++
- intel mkl

# how to built this program:
make sure all dependencies are all loaded in cluster(in our cluster, we use the commands in moduleload.sh to load them), then run this command in the main directory
```shell
make
````
Then you will get a executable file names 'main' in the bin directory.
Use `sbatch` to submit task to the cluster. Submit configurations are specified in the `test18.slurm` file
```shell
sbatch test18.slurm
```
After task finished, you can find a test18.o output file, logs can be found there.
