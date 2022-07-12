1. install CUDA

    Please make sure these ENV are available:
    ```shell
    export PATH=/usr/local/cuda/bin/:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
    export CUDA_BIN_PATH=/usr/local/cuda
    export CUDA_PATH=/usr/local/cuda
    ```
    The pathes are where you installed your CUDA toolkits

2. install oneMKL

    select some [options](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)  and download oneMKL

    After you choose these opetions, there will be an installation guide on the same web page.

3. install magma

    We use the [2.5.4](http://icl.utk.edu/projectsfiles/magma/downloads/magma-2.5.4.tar.gz) version magma, newer version may have some incompatibility problem, please choose the same version.

    use `tar -zxvf magma-2.5.4.tar.gz` to unpack the downloaded package.
    there we be a `README` guide and some `make.inc-examples` in the released folder. you can follow those guide provided by magma, or just use the configuration(We used) below:
    ```make
    GPU_TARGET = Volta

    CC        = gcc
    CXX       = g++
    NVCC      = nvcc
    FORT      = gfortran
    ARCH      = ar
    ARCHFLAGS = cr
    RANLIB    = ranlib

    FPIC      = -fPIC
    CFLAGS    = -O3 $(FPIC) -fopenmp -DNDEBUG -DADD_ -Wall -Wno-strict-aliasing -Wshadow -DMAGMA_WITH_MKL
    FFLAGS    = -O3 $(FPIC)          -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument
    F90FLAGS  = -O3 $(FPIC)          -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument -x f95-cpp-input
    NVCCFLAGS = -O3                  -DNDEBUG -DADD_ -Xcompiler "$(FPIC) -Wall -Wno-unused-function -Wno-strict-aliasing" -std=c++11
    LDFLAGS   =     $(FPIC) -fopenmp

    CXXFLAGS := $(CFLAGS) -std=c++11
    CFLAGS   += -std=c99

    LIB       = -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lstdc++ -lm -lgfortran
    LIB      += -lcublas -lcusparse -lcudart -lcudadevrt

    MKLROOT = /opt/intel/oneapi/mkl/2022.1.0
    CUDADIR = /usr/local/cuda
    -include make.check-mkl
    -include make.check-cuda

    LIBDIR    = -L$(CUDADIR)/lib64 \
                -L$(MKLROOT)/lib/intel64

    INC       = -I$(CUDADIR)/include \
                -I$(MKLROOT)/include
    ```
    Create a `make.inc` file in the main directory, then copy this configuration to `make.inc` you created (Recheck the pathes please).

    Then Run:
    ```shell
    make lib -j
    sudo make install prefix=/usr/local/magma -j
    make clean
    ```
    magma will be installed in `/usr/local/magma`. Bythe way, you may need to set this ENV:
    ```
    export LD_LIBRARY_PATH=/usr/local/magma/lib:$LD_LIBRARY_PATH
    ```


