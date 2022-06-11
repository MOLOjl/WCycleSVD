#include <mpi.h>
#include <iostream>
#include <string>
#include "magma_svd.h"
#include "our_svd.h"

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

using namespace std;

int get_matrix(const char * matrix_path, double* matrix, int h, int w){
    FILE* fp = fopen(matrix_path, "r");
    if(fp==NULL){
        printf("open file falied\n");
        return 0;
    }
    for(int i=0; i < h*w; i++){
        fscanf(fp, "%lf", &matrix[i]);
    }
    return 1;
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
     MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int dev_count = 0;
    hipGetDeviceCount(&dev_count);
    printf("available device:%d\n", dev_count);
    hipSetDevice(world_rank%4);

    int height=512, width=512;
    int shape[3] = {1, height, width};
    string matrix_path = "/public/home/ictapp_x/pyf_folder/comtest/svd_test18/data_in/input.txt";
    
    double *host_A, *host_S, *host_U, *host_V;
    host_A = (double*)malloc(sizeof(double)*height*width);
    host_S = (double*)malloc(sizeof(double)*height);
    host_U = (double*)malloc(sizeof(double)*height*height);
    host_V = (double*)malloc(sizeof(double)*width*width);


    if(get_matrix(matrix_path.data(), host_A, height, width))
    {
        svd_large_matrix(host_A, shape, host_S, host_U, host_V);
    }
    
    magma_svd(1, 512, 512, world_rank%4);

    // Get the name of the processor
    // char processor_name[MPI_MAX_PROCESSOR_NAME];
    // int name_len;
    // MPI_Get_processor_name(processor_name, &name_len);
    // Print off a hello world message
    // printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);

    // Finalize the MPI environment.
    MPI_Finalize();
}
