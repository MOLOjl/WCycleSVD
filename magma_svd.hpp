#include <iostream>
#include "magma_d.h"
#include "magma.h"
#include <string>
#include <chrono>
#include <vector>

using namespace std;
using namespace std::chrono;

// return time
double magma_svd(const char* file_path, int batch, int height, int width, int index=0){
    int minmn = height > width ? width : height;
    
    // printf("svd %d,%dx%d\n", batch, height, width);

    double* host_A = (double*)malloc(sizeof(double) * height * width * batch);
    double* host_S = (double*)malloc(sizeof(double) * minmn);
    double* host_U = (double*)malloc(sizeof(double) * height * height);
    double* host_V = (double*)malloc(sizeof(double) * width * width);

    // string matrix_path1 = "/public/home/ictapp_x/pyf_folder/comtest/svd_test18/data_in/input.txt";

    // read in host A
    FILE* A_fp = fopen(file_path, "r");
    if(A_fp==NULL){
        printf("open file: %s failed!\n", file_path);
        return 0;
    }
    for(int i=0; i < height*width; i++){
        fscanf(A_fp, "%lf", &host_A[i]);
    }
    for(int i=1;i<batch;i++){
        memcpy(&host_A[i*height*width], host_A, height*width*sizeof(double));
    }

    fclose(A_fp);
    double* workspace = (double*)malloc(sizeof(double) * 1);
    int info = 0;
    
    // printf("hi\n");

    steady_clock::time_point t1 = steady_clock::now();

    magma_init();
    magma_device_t devs[10];
    int num_dev=0;
    magma_getdevices(devs, 10, &num_dev);
    printf("available magma device num:%d\n", num_dev);
    if(index < num_dev){
        magma_setdevice(devs[index]);
        // magma_device_t c_dev;
        // magma_getdevice(&c_dev);
        // printf("current device:%d\n", c_dev);     
    }
    else{
        printf("no capable device %d\n", index);
    }

    // query workspace size
    magma_dgesvd(MagmaSomeVec, MagmaSomeVec, height, width, host_A, height, host_S, host_U, height, host_V, width, workspace, -1, &info);
    
    int lwork = (int)workspace[0];
    free(workspace);
    workspace = (double*)malloc(sizeof(double) * lwork);

    for(int iter=0; iter<batch; iter++){
        magma_dgesvd(MagmaSomeVec, MagmaSomeVec, height, width, &host_A[iter*height*width], height, host_S, host_U, height, host_V, width, workspace, lwork, &info); // must be host memory
        hipDeviceSynchronize();
    }

    steady_clock::time_point t2 = steady_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    // printf("%dx%dx%d, magma svd time:%lfs\n", batch, height, width, time_span.count());
    hipDeviceReset();
    free(host_A);free(host_S);free(host_U);free(host_V);free(workspace);

    return time_span.count();
}
