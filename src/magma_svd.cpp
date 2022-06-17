#include <iostream>
#include "magma_d.h"
#include "magma.h"
#include <string>
#include <chrono>
#include <vector>

using namespace std;
using namespace std::chrono;

// return time
double magma_svd(double* host_A, int* shape, double* host_diag, double* host_U, double* host_V, int dev_idx){
    int batch = shape[0]; // ===1
    int height = shape[1];
    int width = height;
    
    int minmn = height;
    
    // printf("svd %d,%dx%d\n", batch, height, width);

    double* host_S = host_diag;

    double* workspace = (double*)malloc(sizeof(double) * 1);
    int info = 0;

    magma_init();
    magma_device_t devs[10];
    int num_dev=0;
    magma_getdevices(devs, 10, &num_dev);
    // printf("available magma device num:%d\n", num_dev);
    if(dev_idx < num_dev){
        magma_setdevice(devs[dev_idx]);
        // magma_device_t c_dev;
        // magma_getdevice(&c_dev);
        // printf("current device:%d\n", c_dev);        
    }
    else{
        printf("no capable device %d\n", dev_idx);
    }

    // steady_clock::time_point t2 = steady_clock::now();

    // query workspace size
    magma_dgesvd(MagmaSomeVec, MagmaSomeVec, height, width, host_A, height, host_S, host_U, height, host_V, width, workspace, -1, &info);
    
    int lwork = (int)workspace[0];
    free(workspace);
    workspace = (double*)malloc(sizeof(double) * lwork);

    magma_dgesvd(MagmaSomeVec, MagmaSomeVec, height, width, host_A, height, host_S, host_U, height, host_V, width, workspace, lwork, &info); // must be host memory
    hipDeviceSynchronize();

    // steady_clock::time_point t2 = steady_clock::now();
    // duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    // printf("%dx%dx%d, magma svd time:%lfs\n", batch, height, width, time_span.count());
    
    hipDeviceReset();
    free(workspace);
    // free(host_A);free(host_S);free(host_U);free(host_V);

    return 0;
}
