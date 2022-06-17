#include <mpi.h>
#include <iostream>
#include <string>
#include "magma_svd.h"
#include "our_svd.h"
#include <sys/types.h>
#include <dirent.h>
#include <chrono>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#define MATRIX_COUNT 20
#define ALL_LOAD 1500

using namespace std;
using namespace std::chrono;

void get_path_size(string pathes[], int sizes[]){
    // FILE* fp1 = fopen("./data_in/pathes_list.txt", "r");
    FILE* fp2 = fopen("./data_in/sizes_list.txt", "r");
    char* path_buff[50];
    for(int i=0; i<MATRIX_COUNT; i++){
        // pathes[i] = (char *)malloc(sizeof(char)*50);
        // fgets(pathes[i], 50, fp1);
        int size_;
        fscanf(fp2, "%d", &size_);
        sizes[i] = size_;

        pathes[i] = "./data_in/" + to_string(i) + "_" + to_string(size_) + ".txt";
    }
}

int get_matrix(const char * matrix_path, double* matrix, int h, int w){
    FILE* fp = fopen(matrix_path, "r");
    if(fp==NULL){
        printf("open file falied: %s\n", matrix_path);
        return 0;
    }
    // printf("open file success: %s\n", matrix_path);
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
    // printf("available device:%d\n", dev_count);
    hipSetDevice(world_rank%4);

    int magnitude = world_size / 50;
    if(magnitude < 1) 
        magnitude = 1;

    int load = ALL_LOAD / magnitude;
    string matrix_pathes[MATRIX_COUNT];
    int matirx_sizes[MATRIX_COUNT];

    get_path_size(matrix_pathes, matirx_sizes);

    if(world_rank == 0){
        printf("p0's load:%d\n", load);
        // for(int i=0; i<MATRIX_COUNT; i++){
        //     printf("path:%s, size:%d\n", matrix_pathes[i], matirx_sizes[i]);
        // }
    }

    // our svd test
	steady_clock::time_point t1 = steady_clock::now();

    for(int i=0; i<load; i++){
        for(int j = 0; j<MATRIX_COUNT; j++){
            int height=matirx_sizes[j], width=height;
            int shape[3] = {1, height, width};
            
            double *host_A, *host_S, *host_U, *host_V;
            host_A = (double*)malloc(sizeof(double)*height*width);
            host_S = (double*)malloc(sizeof(double)*height);
            host_U = (double*)malloc(sizeof(double)*height*height);
            host_V = (double*)malloc(sizeof(double)*width*width);

            if(get_matrix(matrix_pathes[j].data(), host_A, height, width))
            {
                svd_large_matrix(host_A, shape, host_S, host_U, host_V, world_rank%4);
            }

            free(host_A);free(host_S);free(host_U);free(host_V);
        }
    }

	steady_clock::time_point t2 = steady_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    // printf("our svd time total: %lf\n", time_span.count());
    MPI_Barrier(MPI_COMM_WORLD);

    steady_clock::time_point t3 = steady_clock::now();
    
    // magma svd test
    for(int i=0; i<load; i++){
        for(int j = 0; j<MATRIX_COUNT; j++){
            int height=matirx_sizes[j], width=height;
            int shape[3] = {1, height, width};
            
            double *host_A, *host_S, *host_U, *host_V;
            host_A = (double*)malloc(sizeof(double)*height*width);
            host_S = (double*)malloc(sizeof(double)*height);
            host_U = (double*)malloc(sizeof(double)*height*height);
            host_V = (double*)malloc(sizeof(double)*width*width);

            if(get_matrix(matrix_pathes[j].data(), host_A, height, width))
            {
                magma_svd(host_A, shape, host_S, host_U, host_V, world_rank%4);
            }
            
            free(host_A);free(host_S);free(host_U);free(host_V);
        }
    }

	steady_clock::time_point t4 = steady_clock::now();
    duration<double> time_span1 = duration_cast<duration<double>>(t4 - t3);
    // printf("magma svd time total: %lf\n", time_span1.count());

    // Get the name of the processor
    // char processor_name[MPI_MAX_PROCESSOR_NAME];
    // int name_len;
    // MPI_Get_processor_name(processor_name, &name_len);
    // Print off a hello world message
    // printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, world_rank, world_size);


    double my_time1 = time_span.count();
    double my_time2 = time_span1.count();

    double total_time1 = 0;
    double total_time2 = 0;

    MPI_Allreduce(&my_time1, &total_time1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&my_time2, &total_time2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if(world_rank == 0){
        printf("our svd total time:%lf\n", total_time1/world_size);
        printf("magma svd total time:%lf\n", total_time2/world_size);
    }
    // Finalize the MPI environment.
    MPI_Finalize();
}
