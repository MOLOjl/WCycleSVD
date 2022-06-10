DIR=$(shell pwd)

MKLDIRLIB        = /public/home/ictapp/hx_group/intel/composer_xe_2015.2.164/mkl/lib/intel64
MAGMALIB         = /public/software/mathlib/magma/magma-rocm_3.3_develop/lib
MAGMAINCLUDE         = /public/software/mathlib/magma/magma-rocm_3.3_develop/include
# MPIDIRINCLUDE        = /opt/hpc/software/mpi/hpcx/v2.4.1/gcc-7.3.1/include
# MPIDIRLIB        = /opt/hpc/software/mpi/hpcx/v2.4.1/gcc-7.3.1/lib

CC = hipcc
CFLAGS2 = -mcmodel=large -DHAVE_HIP -g -w -I${DIR_INC} -I${MAGMAINCLUDE}
LDFLAGS = -lmkl_rt -L${MKLDIRLIB} -L/public/software/compiler/rocm/rocm-3.3.0/lib  -lrocblas -L${MAGMALIB} -lmagma

start: svd.cpp
	$(CC) $(CFLAGS2) $(LDFLAGS) -o svd svd.cpp -I$DIR -w
