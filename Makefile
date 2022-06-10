DIR=$(shell pwd)
TC_FLAG=-DUSE_TC

start: src/test.cu
	nvcc src/test.cu -gencode arch=compute_80,code=sm_80 -std=c++11 -o test -I$(DIR)/src $(TC_FLAG) -w -lcusolver -lcublas


