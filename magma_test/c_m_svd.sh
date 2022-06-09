# magma version = 2.5.4
# magma install path = /usr/local/magma
# cuda tookit version = 11.4
# CUDAIR = /usr/local/cuda
# mkl release = 2022.1.0.223
# intel mkl path = /opt/intel/oneapi/mkl/2022.1.0

echo "compile ma_svd.cu:"
echo "nvcc ma_svd.cu -o ma -L/usr/local/magma/lib -lmagma_sparse -lmagma -Xcompiler -fopenmp -L/usr/local/cuda/lib64 -L/opt/intel/oneapi/mkl/2022.1.0/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lstdc++ -lm -lgfortran -lcublas -lcusparse -lcudart -lcudadevrt -I/usr/local/magma/include -DNDEBUG -DADD_ -DMIN_CUDA_ARCH=700 -I/usr/local/cuda/include -I/opt/intel/oneapi/mkl/2022.1.0/include -w"
nvcc ma_svd.cu -o ma -L/usr/local/magma/lib -lmagma_sparse -lmagma -Xcompiler -fopenmp -L/usr/local/cuda/lib64 -L/opt/intel/oneapi/mkl/2022.1.0/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lstdc++ -lm -lgfortran -lcublas -lcusparse -lcudart -lcudadevrt -I/usr/local/magma/include -DNDEBUG -DADD_ -DMIN_CUDA_ARCH=700 -I/usr/local/cuda/include -I/opt/intel/oneapi/mkl/2022.1.0/include -w
echo "over"