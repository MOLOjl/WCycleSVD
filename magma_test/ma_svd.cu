#include <iostream>
#include "magma_d.h"
#include "magma.h"
#include <string>
#include <chrono>
#include <vector>

using namespace std;
using namespace std::chrono;

// return time
double magma_svd(int batch, int height, int width){
    int minmn = height > width ? width : height;

    double* host_A = (double*)malloc(sizeof(double) * height * width * batch);
    double* host_S = (double*)malloc(sizeof(double) * minmn);
    double* host_U = (double*)malloc(sizeof(double) * height * height);
    double* host_V = (double*)malloc(sizeof(double) * width * width);

    string matrix_path1 = "../data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";

    // read in host A
    FILE* A_fp = fopen(matrix_path1.data(), "r");
    if(A_fp==NULL){
        matrix_path1 = "./data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";
        A_fp = fopen(matrix_path1.data(), "r");
        if(A_fp==NULL){
            printf("open file %s falied\n", matrix_path1.data());
            return 0;            
        }
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
    
    steady_clock::time_point t1 = steady_clock::now();
    magma_init();

    magma_init();
    magma_device_t devs[10];
    int num_dev=0;
    magma_getdevices(devs, 10, &num_dev);
    printf("avilable device num:%d\n", num_dev);

    // query workspace size
    magma_dgesvd(MagmaSomeVec, MagmaSomeVec, height, width, host_A, height, host_S, host_U, height, host_V, width, workspace, -1, &info);
    
    int lwork = (int)workspace[0];
    free(workspace);
    workspace = (double*)malloc(sizeof(double) * lwork);

    // printf("svd %d,%dx%d\n", batch, height, width);

    for(int iter=0; iter<batch; iter++){
        magma_dgesvd(MagmaSomeVec, MagmaSomeVec, height, width, &host_A[iter*height*width], height, host_S, host_U, height, host_V, width, workspace, lwork, &info); // must be host memory
        cudaDeviceSynchronize();
    }

    steady_clock::time_point t2 = steady_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    // printf("%dx%dx%d, magma svd time:%lfs\n", batch, height, width, time_span.count());
    cudaDeviceReset();
    free(host_A);free(host_S);free(host_U);free(host_V);free(workspace);

    return time_span.count();
}

int main(int argc, char* argv[]){
    
    if(argc == 2){
        int i;
        sscanf(argv[1], "%d", &i);
        if(i==1){
            // warm up
            magma_svd(100, 16, 16);

            vector<int> shape_array = {8, 16, 24, 32};
            vector<int> batch_array = {10, 100, 500};
            printf("===============magma svd - small matrix==============\n");
            printf("matix shape: 8x8   16x16   24x24   32x32\n");

            for(int i=0; i<batch_array.size(); i++){
                double time_re[4] = {0, 0, 0, 0};
                for(int j=0; j<shape_array.size(); j++){
                    time_re[j] = magma_svd(batch_array[i], shape_array[j], shape_array[j]);
                }
                printf("batch %d: %.5lf %.5lf %.5lf %.5lf\n", batch_array[i], time_re[0], time_re[1], time_re[2], time_re[3]);
            }
        }
        if(i==2){
            vector<int> shape_array = {64, 128, 256, 512, 1024};
            vector<int> batch_array = {10, 100, 500};

            printf("=================magma svd - small matrix================\n");
            printf("matix shape: 64x64   128x128   256x256   512x512   1024x1024\n");

            for(int i=0; i<batch_array.size(); i++){
                double time_re[5] = {0, 0, 0, 0, 0};
                for(int j=0; j<shape_array.size(); j++){
                    time_re[j] = magma_svd(batch_array[i], shape_array[j], shape_array[j]);
                }
                printf("batch %d:    %.4lf   %.4lf   %.4lf    %.4lf     %.4lf\n", batch_array[i], time_re[0], time_re[1], time_re[2], time_re[3], time_re[4]);
            }
        }
        if(i==3){
            int shape_array[7] = {500, 1000, 2000, 4000, 5000, 8000, 10000};

            printf("=================magma svd - 1 batch matrix================\n");
            printf("matix shape: 500x500  1000x1000  2000x2000  4000x4000  5000x5000  8000x8000, 10000x10000\n");

            double time_re[7] = {0, 0, 0, 0, 0, 0, 0};
            for(int j=0; j<7; j++){
                time_re[j] = magma_svd(1, shape_array[j], shape_array[j]);
            }
            printf("time     :    %.4lf   %.4lf    %.4lf     %.4lf    %.4lf     %.4lf     %.4lf\n", time_re[0], time_re[1], time_re[2], time_re[3], time_re[4], time_re[5], time_re[6]);
        }
    }
    else{
        magma_svd(1, 512, 512);
    }
    
    return 0;
}
