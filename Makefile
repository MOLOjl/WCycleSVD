DIR=$(shell pwd)
TC_FLAG=-DUSE_TC

all: standalone

standalone: src/test.cu
	nvcc src/test.cu -gencode arch=compute_80,code=sm_80 -keep -std=c++11 -o test_sa -I$(DIR)/src $(TC_FLAG) -w -lcusolver -lcublas

clean:
	-rm -f $(DIR)/bin/* $(DIR)/lib/* test_sa*
