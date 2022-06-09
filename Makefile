DIR=$(shell pwd)

start: src/test.cu magma_test/ma_svd.cu
	nvcc src/test.cu -o test -I$(DIR)/src -w -lcusolver -lcublas
	nvcc magma_test/ma_svd.cu -o magma_test/ma -L/usr/local/magma/lib -lmagma_sparse -lmagma -Xcompiler -fopenmp -L/usr/local/cuda/lib64 -L/opt/intel/oneapi/mkl/2022.1.0/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lstdc++ -lm -lgfortran -lcublas -lcusparse -lcudart -lcudadevrt -I/usr/local/magma/include -DNDEBUG -DADD_ -DMIN_CUDA_ARCH=700 -I/usr/local/cuda/include -I/opt/intel/oneapi/mkl/2022.1.0/include -w

