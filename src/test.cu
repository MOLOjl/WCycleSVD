#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "matrix_generate.hpp"
#include "large_matrix_svd.cu"
#include "small_matrix_svd.cu"
#include "cusolver_svd.cu"

#define ITERS 1
using namespace std;

// 1 batch larget matrix svd test(500-10000)
void test5(){
    vector<int> shape_array1 = {500, 1000, 2000, 4000, 5000, 8000, 10000};

    for(int iter1=0; iter1<shape_array1.size(); iter1++){
        // height < width
        if(true)
        {
            int batch = 1;
            int height = shape_array1[iter1];
            int width = shape_array1[iter1];
            int shape[3] = {batch, height, width};
            int minmn = height > width ? width : height;
            
            double* host_A = (double*)malloc(sizeof(double) * height * width);

            string matrix_path1 = "./data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";   // in case
            // read in host A
            FILE* A_fp = fopen(matrix_path1.data(), "r");
            if(A_fp==NULL){
                generate_matrix(height, width);
                A_fp = fopen(matrix_path1.data(), "r");
                if(A_fp==NULL){
                    printf("open file falied\n");
                    return ;
                }
            }

            for(int i=0; i < height*width; i++){
                fscanf(A_fp, "%lf", &host_A[i]);
            }

            fclose(A_fp);
            
            // copy to device
            double *dev_A;
            cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);    

            // do svd
            double *dev_U, *dev_V, *dev_diag;
            cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
            cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
            cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);  

            double cusolv_time=0, our_time=0;
            for (int I=0; I<ITERS; ++I)
            {
                for(int i=0; i<batch; i++){
                    cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
                }

                double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
                test_result[0] = 2.0;
                
                svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);

                cusolver_svd(dev_A, shape, dev_diag, dev_U, dev_V, test_result);

                cusolv_time += test_result[2];
                our_time += test_result[1];
            }
            cusolv_time /= ITERS;
            our_time /= ITERS;
            printf("matrix:%dx%dx%d, speedup over cusolver: %lf/%lf = %lf\n", batch, height, width, cusolv_time, our_time, cusolv_time / our_time);
            free(host_A);
            cudaFree(dev_A);
            cudaFree(dev_U);
            cudaFree(dev_V);
            cudaFree(dev_diag);
            cudaDeviceReset();
        }           
        printf("====================================\n");
    }
}

// Comparison with cuSOLVER using different batch sizes
void test6(){
    vector<int> shape_array1 = {64, 128, 256, 512, 1024};
    vector<int> batch_array = {10, 100, 500};

    for(int iter1=0; iter1<shape_array1.size(); iter1++){
        for(int iter2=0; iter2<batch_array.size(); iter2++){          
            int batch = batch_array[iter2];
            int height = shape_array1[iter1];
            int width = height;
            int shape[3] = {batch, height, width};
            int minmn = height > width ? width : height;
            
            double* host_A = (double*)malloc(sizeof(double) * height * width);

            string matrix_path1 = "./data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";
            // read in host A
            FILE* A_fp = fopen(matrix_path1.data(), "r");

            if(A_fp==NULL){
                generate_matrix(height, width);
                A_fp = fopen(matrix_path1.data(), "r");
                if(A_fp==NULL){
                    printf("open file falied\n");
                    return ;
                }
            }
            for(int i=0; i < height*width; i++){
                fscanf(A_fp, "%lf", &host_A[i]);
            }

            fclose(A_fp);
            
            // copy to device
            double *dev_A;
            cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);

            // do svd
            double *dev_U, *dev_V, *dev_diag;
            cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
            cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
            cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

            double cusolv_time = 0, our_time = 0;
            for (int I=0; I<ITERS; I++)
            {
                for(int i=0; i<batch; i++){
                    cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
                }                    

                double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
                test_result[0] = 2.0;
                
                svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);
                cusolver_svd(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
                cusolv_time += test_result[2];
                our_time += test_result[1];
            }

            cusolv_time /= ITERS;
            our_time /= ITERS;

            printf("matrix:%dx%dx%d, speedup over cusolver: %lf/%lf = %lf\n", batch, height, width, cusolv_time, our_time, cusolv_time/our_time); 

            free(host_A);
            cudaFree(dev_A);
            cudaFree(dev_U);
            cudaFree(dev_V);
            cudaFree(dev_diag);
            cudaDeviceReset();
        }
        printf("====================================\n");
    }
}

// generate matrix set
void before_test(){
    int h_array[5] = {64, 128, 256, 512, 1024};
    int w_array[5] = {64, 128, 256, 512, 1024};

    int sq_array[7] = {500, 1000, 2000, 4000, 5000, 8000, 10000};

    for(int i=0; i<5; i++){
        printf("generate matrix %d × %d             \r", h_array[i], w_array[i]);
        fflush(stdout);
        generate_matrix(h_array[i], w_array[i]);
    }

    for(int j=0; j<7; j++){
        printf("generate matrix %d × %d             \r", sq_array[j], sq_array[j]);
        fflush(stdout);
        generate_matrix(sq_array[j], sq_array[j]);
    }
    printf("\n over \n");
}

int main(int argc, char* argv[]){
    cudaSetDevice(1);
    if(argc == 2){
        int i;
        sscanf(argv[1], "%d", &i);
        if(i == 5){
            printf("================== test5 ==================\n");
            test5();
        }
        if(i == 6){
            printf("================== test6 ==================\n");
            test6();
        }
        if(i == 16){
            printf("================== test16 ==================\n");
            printf("================== 0.5k-10k matrix ==================\n");
            test5();
            printf("================== 64-1024 matrix ==================\n");
            test6();
        }
        if(i == 99){
            printf("================== generate matrix ==================\n");
            before_test();
        }     
    }
    return 0;
}