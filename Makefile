DIR=$(shell pwd)

start: src/matrix_generate.cpp src/test.cu
	g++ src/matrix_generate.cpp -o bin/mg -w
	nvcc src/test.cu -o bin/test -I$(DIR)/src -w -lcusolver


clean:
	-rm -f $(DIR)/bin/* $(DIR)/lib/*
