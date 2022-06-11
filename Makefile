DIR_INC = ./include
DIR_SRC = ./src
DIR_OBJ = ./obj
DIR_BIN = ./bin

MPIDIRLIB        = /opt/hpc/software/mpi/hpcx/v2.4.1/gcc-7.3.1/lib
MKLDIRLIB        = /public/home/ictapp/hx_group/intel/composer_xe_2015.2.164/mkl/lib/intel64
MAGMALIB         = /public/software/mathlib/magma/magma-rocm_3.3_develop/lib
MPIDIRINCLUDE        = /opt/hpc/software/mpi/hpcx/v2.4.1/gcc-7.3.1/include
MAGMAINCLUDE         = /public/software/mathlib/magma/magma-rocm_3.3_develop/include

# wildcard 把指定目录./src下的所有后缀是.c的文件全部展开 SRC = ./src/1.c ./src/2.c
SRC = $(wildcard ${DIR_SRC}/*.cpp)

# patsubst 将./src下的所有后缀是c的文件全部该成./obj目录下的.o文件 OBJ = ./obj/1.o ./obj/2.o
OBJ = $(patsubst %.cpp,${DIR_OBJ}/%.o,$(notdir ${SRC}))
TARGET = main
BIN_TARGET = ${DIR_BIN}/${TARGET}
CC = hipcc

# -g gdb调试选项，-Wall 一些关键编译打印信息， -I./inc 让c文件的头文件包含路径到./inc目录下去找
#CFLAGS1 = -lmkl_rt -mcmodel=medium -g -w -I${DIR_INC}
CFLAGS2 = -mcmodel=large -DHAVE_HIP -g -w -I${DIR_INC} -I${MPIDIRINCLUDE} -I${MAGMAINCLUDE}
LDFLAGS = -lmkl_rt -DHAVE_HIP -L${MKLDIRLIB} -lmpi -L${MPIDIRLIB} -L/public/software/compiler/rocm/rocm-3.3.0/lib  -lrocblas -L${MAGMALIB} -lmagma
${BIN_TARGET}:${OBJ}
	$(CC) $(LDFLAGS) ./obj/*.o -o $@
${DIR_OBJ}/%.o:${DIR_SRC}/%.cpp
	$(CC) $(CFLAGS2) -c $< -o $@

.PHONY:clean
clean:
	find ${DIR_OBJ} -name *.o -exec rm -rf {} \;
