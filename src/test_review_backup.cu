#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>

#include "matrix_generate.hpp"
#include "large_matrix_svd.cu"
#include "small_matrix_svd.cu"
#include "cusolver_svd.cu"

using namespace std;
using namespace std::chrono;

double get_residual(double* host_A, double* dev_U, double* dev_diag, double* dev_V, int size){
    double* host_U = (double*)malloc(sizeof(double)*size*size);
    double* host_diag = (double*)malloc(sizeof(double) * size);
    double* host_V = (double*)malloc(sizeof(double)*size*size);

    cudaMemcpy(host_U, dev_U, sizeof(double)* size*size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_diag, dev_diag, sizeof(double) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_V, dev_V, sizeof(double)* size*size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    double* host_cA;
    host_cA = (double*)malloc(sizeof(double) * size * size);
    memset(host_cA, 0, 512*512*sizeof(double));

    // host_cA = UDV^T
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double tt = 0;
            for (int z = 0; z < size; z++) {
                tt += host_U[z * size + i] * host_diag[z] * host_V[z * size + j];
            }
            host_cA[j*size + i] = tt;
        }
    }
    
    // get A_Fnorm
    double A_Fnorm = 0;
    for(int i=0; i<size*size; i++){
        A_Fnorm += host_A[i] * host_A[i];
    }
    A_Fnorm = sqrt(A_Fnorm);
    // printf("A_fnorm:%lf\n", A_Fnorm);

    // residual = || host_cA - A || / || A ||
    double residual = 0;
    for(int i=0; i<size*size; i++){
        host_cA[i] -= host_A[i];
        residual += host_cA[i] * host_cA[i];
    }
    residual = sqrt(residual) / A_Fnorm;
    return residual;
}

// TODO: remove this
double get_residual2(double* dev_diag_base, double* dev_diag, int size){
    vector<double> host_diag(size);
    vector<double> host_diag_base(size);

    cudaMemcpy(host_diag.data(), dev_diag, sizeof(double) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_diag_base.data(), dev_diag_base, sizeof(double) * size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    sort(host_diag.begin(), host_diag.end(), greater<double>());
    sort(host_diag_base.begin(), host_diag_base.end(), greater<double>());

    double F_norm = 0;
    for(int i=0; i<size; i++){
        F_norm += host_diag_base[i] * host_diag_base[i];
    }
    
    F_norm = sqrt(F_norm);

    double residual = 0;
    double t = 0;
    for(int i=0; i<size; i++){
        t = host_diag_base[i] - host_diag[i];
        residual += t * t;
    }

    residual = sqrt(residual) / F_norm;
    return residual;
}

// (one side jacobi)svd A_ij vs. (two side jacobi)evd B_ij
void test1();
// tile_w(16-96) affect time
void test2(int iter_in);
// 256 512 matrix Brute-Force search the optimal th-tw
void test3();
// W-cycle SVD for improvement over cuSOLVER (small matrix)
void test4();
// 1 batch larget matrix svd test(500-10000)
void test5();
// Comparison with cuSOLVER using different batch sizes
void test6();
// test auto tuning stratgy's improve
void test7();
void test8();
void test9();
// florida matrix set test, over cusolver
void test10();
// cut_H gemm vs. not cut_H gemm
void test11();
// nvprof 8-1024 ours svd
void test12();
// nvprof 8-1024 cusolver svd
void test13();
// magma baseline 1
void test14();
// magma baseline 2
void test15();
// 64-1024 matrix Brute-Force search the optimal th-tw
void test16();
// accuracy and iters
void test17();


void test1(){
    int h_array[2] = {32, 32};
    int w_array[2] = {512, 256};
    double time_result[8];
    for(int iter1=0; iter1<2; iter1++){
        for(int iter2=0; iter2<2; iter2++){
            int batch = 1;
            int height = h_array[iter1];
            int width = w_array[iter2];
            int minmn = height > width ? width : height;
            int shape[3] = {batch, height, width};
            
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
                // TODO: fix this 
                if(iter1 == 1 && i % 32 >= 16)
                    host_A[i] = 0;
                else
                    fscanf(A_fp, "%lf", &host_A[i]);
            }
            fclose(A_fp);

            int th = height;
            int tw = height;
            double *dev_A;
            cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
            for(int i=0; i<batch; i++){
                cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
            }

            double *dev_U, *dev_V, *dev_diag;
            cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
            cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
            cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

            double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
            test_result[0] = 3.0;
            svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, th, tw, test_result);
            time_result[iter1*2+iter2] =  test_result[1];

            test_result[0] = 6.0;
            svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, th, tw, test_result);
            time_result[4 + iter1*2+iter2] =  test_result[1];

            cudaFree(dev_A);
            cudaFree(dev_U);
            cudaFree(dev_V);
            cudaFree(dev_diag);
            cudaDeviceReset();
        }
    }
    printf("size:     32×256  32×512  16×256  16×512\n");
    printf("SVD A_ij: %.3lf   %.3lf   %.3lf   %.3lf\n", time_result[1], time_result[0], time_result[3], time_result[2]);
    printf("EVD B_ij: %.3lf   %.3lf   %.3lf   %.3lf\n", time_result[5], time_result[4], time_result[7], time_result[6]);
    return;
}

void test2(int iter_in = 0){
    int tw_array[5] = {16, 32, 48, 64, 96};
    int sweeps[5] = {0, 0, 0, 0, 0};
    double result[5] = {0, 0, 0, 0, 0};
    int batch = 100;
    int height = 1536;
    int width = 1536;
    int minmn = height > width ? width : height;
    int shape[3] = {batch, height, width};
    
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

    int th=32;
    int tw=tw_array[iter_in];
    double *dev_A;
    cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
    for(int i=0; i<batch; i++){
        cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
    }

    double *dev_U, *dev_V, *dev_diag;
    cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
    cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
    cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

    double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
    test_result[0] = 4.0;
    svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, th, tw, test_result);

    result[iter_in] =  test_result[1];
    sweeps[iter_in] = (int)test_result[3];

    cudaFree(dev_A);
    cudaFree(dev_U);
    cudaFree(dev_V);
    cudaFree(dev_diag);
    cudaDeviceReset();

    free(host_A);
    // printf result
    printf("w= %d: sweeps: %d, rotations:%d, svd time:%lf\n", tw_array[iter_in], sweeps[iter_in], 2*(width / tw_array[iter_in])-1, result[iter_in]);
}

void test3(){
    int tw_array[4] = {8, 16, 32, 48};
    int th_array[5] = {32, 64, 128, 256, 512};
    double time_result[20];

    // 256*256
    if(true){
        int batch = 100;
        int height = 256;
        int width = 256;
        int minmn = height > width ? width : height;
        int shape[3] = {batch, height, width};
        
        // load A
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

        for(int iter1=0; iter1<4; iter1++){
            for(int iter2=0; iter2<4; iter2++){
                int th=th_array[iter2];
                int tw=tw_array[iter1];
                double *dev_A;
                cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
                for(int i=0; i<batch; i++){
                    cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
                }

                double *dev_U, *dev_V, *dev_diag;
                cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
                cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
                cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

                double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
                test_result[0] = 5.0;
                svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, th, tw, test_result);

                time_result[iter1*4 + iter2] =  test_result[1];

                cudaFree(dev_A);
                cudaFree(dev_U);
                cudaFree(dev_V);
                cudaFree(dev_diag);

                cudaDeviceReset();        
            }
        }
    }

    // print result 1
    printf("===========================256*256===========================\n");
    printf("= th     %d    %d    %d    %d\n", th_array[0], th_array[1], th_array[2], th_array[3]);
    for(int i=0; i<4; i++){
        printf("= tw %d : %.2lf  %.2lf  %.2lf  %.2lf\n", tw_array[i], time_result[i*4 + 0], time_result[i*4 + 1], time_result[i*4 + 2], time_result[i*4 + 3]);
    }

    // 512*512
    if(true){
        int batch = 100;
        int height = 512;
        int width = 512;
        int minmn = height > width ? width : height;
        int shape[3] = {batch, height, width};
        
        // load A
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

        for(int iter1=0; iter1<4; iter1++){
            for(int iter2=0; iter2<5; iter2++){
                int th=th_array[iter2];
                int tw=tw_array[iter1];
                double *dev_A;
                cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
                for(int i=0; i<batch; i++){
                    cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
                }

                double *dev_U, *dev_V, *dev_diag;
                cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
                cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
                cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

                double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
                test_result[0] = 5.0;
                svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, th, tw, test_result);

                time_result[iter1*5 + iter2] =  test_result[1];

                cudaFree(dev_A);
                cudaFree(dev_U);
                cudaFree(dev_V);
                cudaFree(dev_diag);

                cudaDeviceReset();        
            }
        }
    }

    // print result 2
    printf("===========================512*512===========================\n");
    printf("= th     %d    %d    %d    %d    %d\n", th_array[0], th_array[1], th_array[2], th_array[3], th_array[4]);
    for(int i=0; i<4; i++){
        printf("= tw %d : %.2lf  %.2lf  %.2lf  %.2lf  %.2lf\n", tw_array[i], time_result[i*4 + 0], time_result[i*4 + 1], time_result[i*4 + 2], time_result[i*4 + 3], time_result[i*4 + 4]);
    }

    return;
}

void test4(){
    vector<int> shape_array1 = {8, 16, 24, 32};
    vector<int> shape_array2 = {32};
    vector<int> batch_array = {100, 500, 1000, 5000};

    for(int iter1=0; iter1<shape_array1.size(); iter1++){
        for(int iter2=0; iter2<shape_array2.size(); iter2++){
            for(int iter3=0; iter3<batch_array.size(); iter3++){
                // height < width
                if(true)
                {
                    int batch = batch_array[iter3];
                    int height = shape_array1[iter1];
                    int width = shape_array2[iter2];
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

                    for(int i=0; i<batch; i++){
                        cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
                    }

                    // do svd
                    double *dev_U, *dev_V, *dev_diag;
                    cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
                    cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
                    cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

                    double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time

                    test_result[0] = 2.0;
                    
                    svd_samll_matrix(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
                    cusolver_svd_batched(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
                    printf("matrix:%d×%d×%d, speedup over cusolver: %lf/%lf = %lf\n", batch, height, width, test_result[2], test_result[1], test_result[2]/test_result[1]); 

                    free(host_A);
                    cudaFree(dev_A);
                    cudaFree(dev_U);
                    cudaFree(dev_V);
                    cudaFree(dev_diag);
                    cudaDeviceReset();
                    // break;
                
                }           
                
                if(iter1 != 3)
                {
                    int batch = batch_array[iter3];
                    int width = shape_array1[iter1];
                    int height = shape_array2[iter2];
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
                    

                    for(int i=0; i<batch; i++){
                        cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
                    }

                    // do svd
                    double *dev_U, *dev_V, *dev_diag;
                    cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
                    cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
                    cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

                    double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time

                    test_result[0] = 2.0;
                    svd_samll_matrix(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
                    cusolver_svd_batched(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
                    printf("matrix:%d×%d×%d, speedup over cusolver: %lf/%lf = %lf\n", batch, height, width, test_result[2], test_result[1], test_result[2]/test_result[1]); 

                    free(host_A);
                    cudaFree(dev_A);
                    cudaFree(dev_U);
                    cudaFree(dev_V);
                    cudaFree(dev_diag);
                    cudaDeviceReset();
                
                }
                
                if(iter1 != 3)
                {
                    int batch = batch_array[iter3];
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
                    

                    for(int i=0; i<batch; i++){
                        cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
                    }

                    // do svd
                    double *dev_U, *dev_V, *dev_diag;
                    cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
                    cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
                    cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

                    double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time

                    test_result[0] = 2.0;
                    svd_samll_matrix(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
                    cusolver_svd_batched(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
                    printf("matrix:%d×%d×%d, speedup over cusolver: %lf/%lf = %lf\n", batch, height, width, test_result[2], test_result[1], test_result[2]/test_result[1]); 

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
    }
}

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

            for(int i=0; i<batch; i++){
                cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
            }

            // do svd
            double *dev_U, *dev_V, *dev_diag;
            cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
            cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
            cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

            double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
            test_result[0] = 2.0;
            
            svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);

            cusolver_svd(dev_A, shape, dev_diag, dev_U, dev_V, test_result);

            printf("matrix:%d×%d×%d, speedup over cusolver: %lf/%lf = %lf\n", batch, height, width, test_result[2], test_result[1], test_result[2]/test_result[1]);

            free(host_A);
            cudaFree(dev_A);
            cudaFree(dev_U);
            cudaFree(dev_V);
            cudaFree(dev_diag);
            cudaDeviceReset();
        }           
        printf("====================================\n");
        // break;
    }
}

void test6(){
    vector<int> shape_array1 = {64, 128, 256, 512, 1024};
    vector<int> shape_array2 = {1024};
    vector<int> batch_array = {10};

    for(int iter1=0; iter1<shape_array1.size(); iter1++){
        for(int iter2=0; iter2<shape_array2.size(); iter2++){
            for(int iter3=0; iter3<batch_array.size(); iter3++){
                // height < width
                if(true)
                {
                    int batch = batch_array[iter3];
                    int height = shape_array1[iter1];
                    int width = shape_array2[iter2];
                    int shape[3] = {batch, height, width};
                    int minmn = height > width ? width : height;
                    
                    double* host_A = (double*)malloc(sizeof(double) * height * width);
                    string matrix_path1 = "./data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";
                    
                    // load host A
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
                    

                    for(int i=0; i<batch; i++){
                        cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
                    }

                    // do svd
                    double *dev_U, *dev_V, *dev_diag;
                    cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
                    cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
                    cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

                    double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time

                    test_result[0] = 2.0;
                    svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);
                    cusolver_svd(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
                    printf("matrix:%d×%d×%d, speedup over cusolver: %lf/%lf = %lf\n", batch, height, width, test_result[2], test_result[1], test_result[2]/test_result[1]); 

                    free(host_A);
                    cudaFree(dev_A);
                    cudaFree(dev_U);
                    cudaFree(dev_V);
                    cudaFree(dev_diag);
                    cudaDeviceReset();
                }           
                
                if(iter1 != 4)
                {
                    int batch = batch_array[iter3];
                    int width = shape_array1[iter1];
                    int height = shape_array2[iter2];
                    int shape[3] = {batch, height, width};
                    int minmn = height > width ? width : height;
                    
                    double* host_A = (double*)malloc(sizeof(double) * height * width);

                    string matrix_path1 = "./data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";

                    // load host A
                    FILE* A_fp = fopen(matrix_path1.data(), "r");
                    if(A_fp==NULL){
                        generate_matrix(height, width);
                        A_fp = fopen(matrix_path1.data(), "r");
                        if(A_fp==NULL){
                            printf("open/write file falied\n");
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

                    for(int i=0; i<batch; i++){
                        cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
                    }

                    // do svd
                    double *dev_U, *dev_V, *dev_diag;
                    cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
                    cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
                    cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

                    double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time

                    test_result[0] = 2.0;
                    svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);
                    cusolver_svd(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
                    printf("matrix:%d×%d×%d, speedup over cusolver: %lf/%lf = %lf\n", batch, height, width, test_result[2], test_result[1], test_result[2]/test_result[1]); 

                    free(host_A);
                    cudaFree(dev_A);
                    cudaFree(dev_U);
                    cudaFree(dev_V);
                    cudaFree(dev_diag);
                    cudaDeviceReset();
                
                }
                
                if(iter1 != 4)
                {
                    int batch = batch_array[iter3];
                    int height = shape_array1[iter1];
                    int width = shape_array1[iter1];
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
                    
                    for(int i=0; i<batch; i++){
                        cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
                    }

                    // do svd
                    double *dev_U, *dev_V, *dev_diag;
                    cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
                    cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
                    cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

                    double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time

                    test_result[0] = 2.0;
                    
                    svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);
                    cusolver_svd(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
                    printf("matrix:%d×%d×%d, speedup over cusolver: %lf/%lf = %lf\n", batch, height, width, test_result[2], test_result[1], test_result[2]/test_result[1]); 

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
    }
}

void test7(){
    
    vector<int> size_array = {64, 128, 256, 512, 1024};
    vector<int> size_array1 = {64, 128, 256, 512, 1024};
    vector<int> batch_array = {10, 100, 500};
    
    int s1=0, s2=0;
    for(int iter1=s1; iter1<(int)size_array.size(); iter1++){
        for(int iter2=s2; iter2<(int)batch_array.size(); iter2++){

            int batch = batch_array[iter2];
            int height = size_array[iter1];
            int width = size_array1[iter1];
            int shape[3] = {batch, height, width};
            int minmn = height > width ? width : height;

            double* host_A = (double*)malloc(sizeof(double) * height * width);
            string matrix_path1 = "./data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";
            
            // load host A
            FILE* A_fp = fopen(matrix_path1.data(), "r");
            if(A_fp==NULL){
                generate_matrix(height, width);
                A_fp = fopen(matrix_path1.data(), "r");
                if(A_fp==NULL){
                    printf("open/write file falied\n");
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
            

            for(int i=0; i<batch; i++){
                cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
            }

            // do svd
            double *dev_U, *dev_V, *dev_diag;
            cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
            cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
            cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

            
            double temp[3] = {1.0, 1.0, 1.0};
            double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time

            test_result[0] = 0;
            svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);
            temp[0] = test_result[1];

            test_result[0] = 1.0;
            // svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);
            temp[1] = test_result[1];

            test_result[0] = 2.0;
            svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);
            temp[2] = test_result[1];

            
            printf("matrix:%d×%d×%d, over no tailoring:\n", batch, height, width);
            // printf("tailoring on expertise speedup: %lf / %lf = %lf\n", temp[0], temp[1], temp[0]/temp[1]);
            printf("tailoring on Auto-tuning speedup: %lf / %lf = %lf\n", temp[0], temp[2], temp[0]/temp[2]);
            printf("==================================================================\n");

            free(host_A);
            cudaFree(dev_A);
            cudaFree(dev_U);
            cudaFree(dev_V);
            cudaFree(dev_diag);
            cudaDeviceReset();
        }
        // break;
    }
    return;
}

// TODO: (over)
void test8(){
    int size_array[4] = {100, 128, 256, 512};
    for(int iter=0; iter<4; iter++){
        int batch = 200;
        int height = size_array[iter];
        int width = size_array[iter];
        int th=0, tw=0;
        int shape[3] = {batch, height, width};
        int minmn = height > width ? width : height;

        printf("matrix:%d×%d×%d\n", batch, height, width);

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

        double *dev_A;
        cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
        for(int i=0; i<batch; i++){
            cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
        }

        double *dev_U, *dev_V, *dev_diag;
        cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
        cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
        cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

        double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
        test_result[0] = 2.0;

        // our svd
        if(height <= 48 && width <= 48){
            svd_samll_matrix(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
        }
        else{
            svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, th, tw, test_result);
        }
        
        printf("w-cycle svd time:%lf\n", test_result[1]);

        cudaMemset(dev_V, 0, sizeof(double) * width * width * batch);
        cudaMemset(dev_U, 0, sizeof(double) * height * height * batch);
        cudaMemset(dev_diag, 0, sizeof(double) * minmn * batch);


        // cusolver svd
        if(height <= 32 && width <= 32){
            cusolver_svd_batched(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
        }
        else{
        cusolver_svd(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
        }
        
        printf("cuslover svd time:%lf\n", test_result[2]); 
        printf("===========================\n");
        free(host_A);
        cudaFree(dev_A);
        cudaFree(dev_U);
        cudaFree(dev_V);
        cudaFree(dev_diag);
        cudaDeviceReset();

    }
    return;
}

void test9(){
    int size_array[4] = {128, 256, 512, 1024};
    int th_array[6] = {32, 1, 32, 1, 32, 0};
    int tw_array[6] = {8, 8, 48, 48, 32, 0};
    
    double time_result[24];
    int schedule = 1;
    for(int iter1=0; iter1<4; iter1++){
        int batch = 100;
        int height = size_array[iter1];
        int width = size_array[iter1];
        int minmn = height > width ? width : height;
        int shape[3] = {batch, height, width};
        
        // load A
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

        for(int iter2=0; iter2<6; iter2++){
            int th=th_array[iter2];
            int tw=tw_array[iter2];
            if(th == 1)   th=height;
            
            double *dev_A;
            cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
            for(int i=0; i<batch; i++){
                cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
            }

            double *dev_U, *dev_V, *dev_diag;
            cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
            cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
            cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

            double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
            test_result[0] = 5.0;
            svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, th, tw, test_result);
            
            time_result[iter2*4 + iter1] = test_result[1];

            cudaFree(dev_A);
            cudaFree(dev_U);
            cudaFree(dev_V);
            cudaFree(dev_diag);

            cudaDeviceReset();
            schedule ++;
            printf("dealing... %d/24 \r", schedule);
            fflush(stdout);
        }
    }

    // print result
    printf("\nsize   %d    %d    %d    %d\n", size_array[0], size_array[1], size_array[2], size_array[3]);
    for(int i=0; i<6; i++){
        if(th_array[i]==1){
            printf("%s-%d: %.3lf  %.3lf  %.3lf  %.3lf\n", "m", tw_array[i]/2, time_result[i*4+0], time_result[i*4+1], time_result[i*4+2], time_result[i*4+3]);
        }
        else if(th_array[i]==0){
            printf("auto : %.3lf  %.3lf  %.3lf  %.3lf\n", time_result[i*4+0], time_result[i*4+1], time_result[i*4+2], time_result[i*4+3]);
        }
        else{
            printf("%d-%d: %.3lf  %.3lf  %.3lf  %.3lf\n", th_array[i], tw_array[i]/2, time_result[i*4+0], time_result[i*4+1], time_result[i*4+2], time_result[i*4+3]);
        }
    }
    return;
}

#define RANK_COUNT 5
#define MAX_BATCH 301
void test10(){
    int matrix_shapes[5] = {32, 64, 128, 256, 512};

    for(int i=1; i<=5; i++){
        string folder_path1 = "./data/UF_matrixset/" + to_string(i) + "/";

        double* host_A = (double*)malloc(sizeof(double)*MAX_BATCH * matrix_shapes[i-1]*matrix_shapes[i-1]);
        int probe = 0;
        int j = 0; // matrix count
        while(true){
            j ++;       
            string matrix_path1 = folder_path1 + to_string(j) + ".txt";

            FILE* matrix_fp = fopen(matrix_path1.data(), "r");
            if(matrix_fp == NULL){
                    break;
            }

            while(fscanf(matrix_fp, "%lf", &host_A[probe]) != EOF && probe < MAX_BATCH * matrix_shapes[i-1]*matrix_shapes[i-1])
                probe ++;

            fclose(matrix_fp);
            if(j == MAX_BATCH)
                break;
        }

        int batch = j-1;
        if(batch==0){
            printf("no matrix loaded\n");
            return;
        }
            

        int height = matrix_shapes[i-1];
        int width = matrix_shapes[i-1];
        int shape[3] = {j-1, height, width};
        int minmn = height > width ? width : height;

        // copy to device
        double* dev_A;
        cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
        cudaMemcpy(dev_A, host_A, sizeof(double) * height * width * batch, cudaMemcpyHostToDevice);

        double *dev_U, *dev_V, *dev_diag;
        cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
        cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
        cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);
        
        // do test
        double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
        test_result[0] = 2.0;

        if(matrix_shapes[i-1] <= 32){
            svd_samll_matrix(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
            cusolver_svd_batched(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
        }    
        else{
            svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);
            cusolver_svd(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
        }

        printf("matrix: %d×%d×%d, speedup over cusolver: %lf / %lf = %lf\n", batch, height, width, test_result[2], test_result[1], test_result[2]/test_result[1]);
        printf("=========================================================\n");
    }
}

void test11(){
    int shape_array[5] = {64, 128, 256, 512, 1024};
    int batch_array[3] = {10, 100, 500};
    for(int iter1=0; iter1<5; iter1++){
        for(int iter2=0; iter2<3; iter2++){
            int batch = batch_array[iter2];
            int height = shape_array[iter1];
            int width = shape_array[iter1];
            int th=0, tw=0;
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

            double *dev_A;
            cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
            for(int i=0; i<batch; i++){
                cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
            }

            double *dev_U, *dev_V, *dev_diag;
            cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
            cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
            cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

            double test_result[4] = {7.0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
            
            double result_buff1[6] = {0,0,0,0,0,0};
            double result_buff2[6] = {0,0,0,0,0,0};
            int deep = 0;

            for(th=32; th<=height; th*=2){
                svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, th, 32, test_result);
                result_buff1[deep] = test_result[3];    // gemm time
                result_buff2[deep] = test_result[2];    // evd and update time
                deep ++;
            }

            deep = 0;   // reset
            printf("==batch: %d, matrix: %d x %d==\n", batch, height, width);
            printf("tile_h             32        64        128       256       512       1024\n");
            printf("GEMM time      : ");
            for(th=32; th<=1024; th*=2)
            {
                printf("%.6lf  ", result_buff1[deep]);
                deep ++;
            }
            deep = 0;
            printf("\nEVD&Update time: ");
            for(th=32; th<=1024; th*=2)
            {
                printf("%.6lf  ", result_buff2[deep]);
                deep ++;
            }
            printf("\n");
            free(host_A);
            cudaFree(dev_A);
            cudaFree(dev_U);
            cudaFree(dev_V);
            cudaFree(dev_diag);
            cudaDeviceReset();
        }
    }
}

void test12(int batch, int shape_){

    int height = shape_;
    int width = shape_;
    int th=0, tw=0;
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

    double *dev_A;
    cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
    for(int i=0; i<batch; i++){
        cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
    }

    double *dev_U, *dev_V, *dev_diag;
    cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
    cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
    cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

    double test_result[4] = {5.0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
    if(height<=48)
        svd_samll_matrix(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
    else
        svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);

    printf("case %dx%dx%d, our svd time:%lf\n", batch, height, width, test_result[1]);
    free(host_A);
    cudaFree(dev_A);
    cudaFree(dev_U);
    cudaFree(dev_V);
    cudaFree(dev_diag);
    cudaDeviceReset();
}

void test13(int batch, int shape_){
    // int shape_array[5] = {64, 128, 256, 512, 1024};
    // int batch_array[3] = {10, 100, 500};

    int height = shape_;
    int width = shape_;
    int th=0, tw=0;
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

    double *dev_A;
    cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
    for(int i=0; i<batch; i++){
        cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
    }

    double *dev_U, *dev_V, *dev_diag;
    cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
    cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
    cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

    double test_result[4] = {5.0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
    if(height<=32)
        cusolver_svd_batched(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
    else
        cusolver_svd(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
    printf("case %dx%dx%d, cusolver svd time:%lf\n", batch, height, width, test_result[2]);
    
    free(host_A);
    cudaFree(dev_A);
    cudaFree(dev_U);
    cudaFree(dev_V);
    cudaFree(dev_diag);
    cudaDeviceReset();
}

void test14(){
    vector<int> shape_array1 = {500, 1000, 2000, 4000, 5000, 8000, 10000};

    for(int iter1=0; iter1<shape_array1.size() - 2; iter1++){
        // height < width
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
        
        steady_clock::time_point t1 = steady_clock::now();
        
        // copy to device
        double *dev_A;
        cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);      
        cudaMemcpy(dev_A, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);

        // do svd
        double *dev_U, *dev_V, *dev_diag;
        cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
        cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
        cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

        double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
        test_result[0] = 2.0;
        
        svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);

        vector<double> re0(minmn * batch);
        vector<double> re1(height * height * batch);
        vector<double> re2(width * width * batch);
        cudaMemcpy(re0.data(), dev_diag, sizeof(double) * minmn * batch, cudaMemcpyDeviceToHost);
        cudaMemcpy(re1.data(), dev_U, sizeof(double) * height * height * batch, cudaMemcpyDeviceToHost);
        cudaMemcpy(re2.data(), dev_V, sizeof(double) * width * width * batch, cudaMemcpyDeviceToHost);

        steady_clock::time_point t2 = steady_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

        printf("matrix:%d×%d×%d, our svd total: %lf\n", batch, height, width, time_span);

        free(host_A);
        cudaFree(dev_A);
        cudaFree(dev_U);
        cudaFree(dev_V);
        cudaFree(dev_diag);
        cudaDeviceReset();
        // break;
    }
}

void test15(){
    // vector<int> shape_array2 = {8, 16, 24, 32, 64, 128, 256, 512, 1024};
    vector<int> shape_array2 = {32, 64, 128, 256, 512, 1024};
    // vector<int> batch_array = {10, 100, 500};
    vector<int> batch_array = {10};

    for(int iter2=0; iter2<shape_array2.size(); iter2++){
        for(int iter3=0; iter3<batch_array.size(); iter3++){
            int batch = batch_array[iter3];
            int height = shape_array2[iter2];
            int width = height;
            
            int th=0, tw=0;
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

            steady_clock::time_point t1 = steady_clock::now();
            
            double *dev_A;
            cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
            for(int i=0; i<batch; i++){
                cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
            }

            double *dev_U, *dev_V, *dev_diag;
            cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
            cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
            cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

            double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
            test_result[0] = 2.0;

            // our svd
            if(height <= 48 && width <= 48){
                svd_samll_matrix(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
            }
            else{
                svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, th, tw, test_result);
            }
            
            vector<double> re0(minmn * batch);
            vector<double> re1(height * height * batch);
            vector<double> re2(width * width * batch);
            cudaMemcpy(re0.data(), dev_diag, sizeof(double) * minmn * batch, cudaMemcpyDeviceToHost);
            cudaMemcpy(re1.data(), dev_U, sizeof(double) * height * height * batch, cudaMemcpyDeviceToHost);
            cudaMemcpy(re2.data(), dev_V, sizeof(double) * width * width * batch, cudaMemcpyDeviceToHost);

            steady_clock::time_point t2 = steady_clock::now();
            duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

            printf("matrix:%d×%d×%d, our svd total: %lf\n", batch, height, width, time_span);

            free(host_A);
            cudaFree(dev_A);
            cudaFree(dev_U);
            cudaFree(dev_V);
            cudaFree(dev_diag);
            cudaDeviceReset();
        }
    }
    printf("======================\n");
}

void test16(){
    vector<int> shape_array = {64, 128, 256, 384, 480, 512, 640, 768, 896, 992, 1024};
    // vector<int> shape_array = {512};
    int tw_array[4] = {8, 16, 32, 48};

    // 32*(1-16), 1024
    vector<vector<double>> time_result(4, vector<double>(20, 0));

    // for(int i=0;i<4;i++)
    //     fill(time_result[i].begin(), time_result[i].end(), 0);

    // 512*512
    for(int i=0; i<shape_array.size(); i++){
        int batch = 100;
        int height = shape_array[i];
        int width = height;
        int minmn = height > width ? width : height;
        int shape[3] = {batch, height, width};
        
        // load A
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

        int tw=0, th=height;
        vector<int> th_array(20, 0);

        th_array[0] = height;
        int deep = 0;
        while (true)
        { 
            if(th == 32 || deep == 19)
                break;
            deep ++; 
            int i = 1;
            while(deep + i < 33){
                th = ceil(height/(deep + i)/32) * 32;
                i++;
                if(th != th_array[deep-1])
                    break;
            }
            th_array[deep] = th; 
        }
        
        for(int iter1=2; iter1<4; iter1++){
            for(int iter2=0; iter2<20; iter2++){
                tw = tw_array[iter1];
                th = th_array[iter2];
                if(th == 0)
                    break;
                double *dev_A;
                cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
                for(int i=0; i<batch; i++){
                    cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
                }

                double *dev_U, *dev_V, *dev_diag;
                cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
                cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
                cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

                double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
                test_result[0] = 5.0;
                svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, th, tw, test_result);

                time_result[iter1][iter2] =  test_result[1];

                cudaFree(dev_A);
                cudaFree(dev_U);
                cudaFree(dev_V);
                cudaFree(dev_diag);

                cudaDeviceReset();
                printf("batch:%d, %dx%d tile:%dx%d done    \r",batch, height, width, th, tw);
                fflush(stdout);
            }
        }

        printf("\nmatrix:%dx%d-----------------------------------------------\n", height, width);
        for(int i=0; i<20; i++){
            if(th_array[i]==0)
                break;
            printf("       %d", th_array[i]);
        }
        printf("\n");
        for(int i=0; i<4; i++){
            if(i==0) 
                printf("tw:%d ", tw_array[i]);
            else
                printf("tw:%d", tw_array[i]);
            for(int j=0; j<20; j++){
                if(th_array[j]==0)
                    break;
                printf(" %lf", time_result[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    return;
}

void test17(){
    string matrixes[10] = {"lp_nug06", "mbeacxc", "Trec10", "bcsstm07", "tols340", "flower_7_1", "impcol_d", "ash331", "robot24c1_mat5", "Harvard500"};
    for(int iter1=9; iter1<10; iter1++){
        printf("======================matrix:%s======================\n", matrixes[iter1].c_str());
        for(int iter2=1; iter2<20; iter2++){
            int batch = 1;
            int height = 512;
            int width = 512;
            int th=0, tw=0;
            int shape[3] = {batch, height, width};
            int minmn = height > width ? width : height;

            double* host_A = (double*)malloc(sizeof(double) * height * width);

            string matrix_path1 = "./data/robust_test/" + matrixes[iter1] + ".txt";

            // read in host A
            FILE* A_fp = fopen(matrix_path1.data(), "r");
            if(A_fp==NULL){
                printf("open file falied\n");
                return ;
            }

            for(int i=0; i < height*width; i++){
                fscanf(A_fp, "%lf", &host_A[i]);
            }

            fclose(A_fp);

            double *dev_A;
            cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
            for(int i=0; i<batch; i++){
                cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
            }

            double *dev_U, *dev_V, *dev_diag;
            cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
            cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
            cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

            double test_result[4] = {9.0, 1.0, 1.0, iter2}; // 0:tag, 1:time
            svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);
         
            free(host_A);
            cudaFree(dev_A);
            cudaFree(dev_U);
            cudaFree(dev_V);
            cudaFree(dev_diag);
            cudaDeviceReset();
        } 
    }
}

// full evd and 1-round evd time(just evd kernel)
void test18(int batch, int shape_){
    int height = shape_;
    int width = shape_;
    int th=0, tw=0;
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

    double *dev_A;
    cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
    for(int i=0; i<batch; i++){
        cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
    }

    double *dev_U, *dev_V, *dev_diag;
    cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
    cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
    cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

    double test_result[4] = {8.0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
    if(height<=48)
        svd_samll_matrix(dev_A, shape, dev_diag, dev_U,dev_V, test_result);
    else
        svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);

    printf("case %dx%dx%d, evd1 time:%lf, evd10 time:%lf\n", batch, height, width, test_result[3], test_result[2]);
    free(host_A);
    cudaFree(dev_A);
    cudaFree(dev_U);
    cudaFree(dev_V);
    cudaFree(dev_diag);
    cudaDeviceReset();
}

// nvprof 8-1024 ours svd
void test19(int batch, int shape_, int th=32, int tw=16){
    int height = shape_;
    int width = shape_;
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

    double *dev_A;
    cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
    for(int i=0; i<batch; i++){
        cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
    }

    double *dev_U, *dev_V, *dev_diag;
    cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
    cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
    cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

    double test_result[4] = {5.0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
    svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, th, tw, test_result);

    printf("case %dx%dx%d, our svd time:%lf\n", batch, height, width, test_result[1]);
    free(host_A);
    cudaFree(dev_A);
    cudaFree(dev_U);
    cudaFree(dev_V);
    cudaFree(dev_diag);
    cudaDeviceReset();
}

// accuracy and iters
void test20(int iter2=15){
    string matrixes[7] = {"lp_nug06", "mbeacxc", "Trec10", "bcsstm07", "tols340", "flower_7_1", "impcol_d"};

    for(int iter1=0; iter1<7; iter1++){
        printf("---------------------matrix: %s-------------------\n", matrixes[iter1].c_str());
        int batch = 1;
        int height = 512;
        int width = 512;
        int th=0, tw=0;
        int shape[3] = {batch, height, width};
        int minmn = height > width ? width : height;

        double* host_A = (double*)malloc(sizeof(double) * height * width);

        string matrix_path1 = "./data/robust_test/" + matrixes[iter1] + ".txt";

        // read in host A
        FILE* A_fp = fopen(matrix_path1.data(), "r");
        if(A_fp==NULL){
            printf("open file falied\n");
            return ;
        }

        for(int i=0; i < height*width; i++){
            fscanf(A_fp, "%lf", &host_A[i]);
        }

        fclose(A_fp);

        double *dev_A, *dev_A0;
        cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
        cudaMalloc((void **)&dev_A0, sizeof(double) * height * width * batch);
        for(int i=0; i<batch; i++){
            cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
        }

        double *dev_U, *dev_V, *dev_diag, *dev_diag_base1, *dev_diag_base2;
        cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
        cudaMalloc((void **)&dev_diag_base1, sizeof(double) * minmn * batch);
        cudaMalloc((void **)&dev_diag_base2, sizeof(double) * minmn * batch);
        cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
        cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

        cudaMemcpy(dev_A0, dev_A, sizeof(double) * height * width * batch, cudaMemcpyDeviceToDevice);
        double test_result[4] = {10.0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
        test_result[3] = 100;
        cusolver_svd(dev_A0, shape, dev_diag_base1, dev_U, dev_V, test_result);
        
        // vector<double> re1(minmn);
        // cudaMemcpy(re1.data(), dev_diag_base, sizeof(double) * minmn, cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize();
        // sort(re1.begin(), re1.end(), greater<double>());
        // print_matrix(re1.data(), 32, 16, "dev_diag_base1.txt");

        test_result[3] = 100;
        svd_large_matrix(dev_A, shape, dev_diag_base2, dev_U, dev_V, 0, 0, test_result);

        // cudaMemcpy(re1.data(), dev_diag, sizeof(double) * minmn, cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize();
        // sort(re1.begin(), re1.end(), greater<double>());
        // print_matrix(re1.data(), 32, 16, "dev_diag_base2.txt");
        
        test_result[3] = iter2;
        svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);

        double residual2 = get_residual2(dev_diag_base2, dev_diag, height);
        printf("our residual:%e\n", residual2);

        cudaMemset(dev_V, 0, sizeof(double) * width * width * batch);
        cudaMemset(dev_U, 0, sizeof(double) * height * height * batch);
        cudaMemset(dev_diag, 0, sizeof(double) * minmn * batch);

        cusolver_svd(dev_A, shape, dev_diag, dev_U, dev_V, test_result);

        double residual3 = get_residual2(dev_diag_base1, dev_diag, height);
        printf("cusolver residual:%e\n", residual3);

        // vector<double> re1(minmn);
        // cudaMemcpy(re1.data(), dev_diag, sizeof(double) * minmn, cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize();
        // sort(re1.begin(), re1.end(), greater<double>());
        // print_matrix(re1.data(), 32, 16, "dev_diag1.txt");

        // cudaMemcpy(re1.data(), dev_diag_base, sizeof(double) * minmn, cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize();
        // sort(re1.begin(), re1.end(), greater<double>());
        // print_matrix(re1.data(), 32, 16, "dev_diag_base.txt");

        free(host_A);
        cudaFree(dev_A);
        cudaFree(dev_U);
        cudaFree(dev_V);
        cudaFree(dev_diag);
        cudaDeviceReset();
    }
}

// accuracy and iters
void test21(int iter2=15){
    string matrixes[7] = {"lp_nug06", "mbeacxc", "Trec10", "bcsstm07", "tols340", "flower_7_1", "impcol_d"};

    for(int iter1=6; iter1<7; iter1++){
        printf("---------------------matrix: %s-------------------\n", matrixes[iter1].c_str());
        int batch = 1;
        int height = 512;
        int width = 512;
        int th=0, tw=0;
        int shape[3] = {batch, height, width};
        int minmn = height > width ? width : height;

        double* host_A = (double*)malloc(sizeof(double) * height * width);

        string matrix_path1 = "./data/robust_test/" + matrixes[iter1] + ".txt";

        // read in host A
        FILE* A_fp = fopen(matrix_path1.data(), "r");
        if(A_fp==NULL){
            printf("open file falied\n");
            return ;
        }

        for(int i=0; i < height*width; i++){
            fscanf(A_fp, "%lf", &host_A[i]);
        }

        fclose(A_fp);

        double *dev_A;
        cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
        for(int i=0; i<batch; i++){
            cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
        }

        double *dev_U, *dev_V, *dev_diag;
        cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
        cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
        cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);
        
        double test_result[4] = {10.0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
        test_result[3] = iter2;
        svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);
        double residual = get_residual(host_A, dev_U, dev_diag, dev_V, height);
        printf("our residual:%e\n", residual);

        // double* re4 = (double*)malloc(sizeof(double) * height * height);
        // cudaMemcpy(re4, dev_U, sizeof(double) * height * height, cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize();
        // print_matrix(re4, 512, 512, "U.txt");        
        // int orth = orth_matrix_verify(re4, height);
        // printf("U orth? %d\n", orth);

        // vector<double> re5(width * width);
        // cudaMemcpy(re5.data(), dev_V, sizeof(double) * width * width, cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize();
        // print_matrix(re5.data(), 512, 512, "U.txt");

        cudaMemset(dev_V, 0, sizeof(double) * width * width * batch);
        cudaMemset(dev_U, 0, sizeof(double) * height * height * batch);
        cudaMemset(dev_diag, 0, sizeof(double) * minmn * batch);
        cusolver_svd(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
        double residual1 = get_residual(host_A, dev_U, dev_diag, dev_V, height);
        printf("cusolver residual:%e\n", residual1);

        // double* re5 = (double*)malloc(sizeof(double) * height * height);
        // cudaMemcpy(re5, dev_U, sizeof(double) * height * height, cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize();
        // print_matrix(re5, 512, 512, "U1.txt");
        
        // int orth1 = orth_matrix_verify(re5, height);
        // printf("U orth? %d\n", orth1);

        free(host_A);
        cudaFree(dev_A);
        cudaFree(dev_U);
        cudaFree(dev_V);
        cudaFree(dev_diag);
        cudaDeviceReset();
    }
}

// accuracy and iters
void test22(){
    string matrixes[10] = {"lp_nug06", "mbeacxc", "Trec10", "bcsstm07", "tols340", "flower_7_1", "impcol_d", "ash331", "robot24c1_mat5", "Harvard500"};
    for(int iter1=9; iter1<10; iter1++){
        printf("======================matrix:%s======================\n", matrixes[iter1].c_str());
        int batch = 1;
        int height = 512;
        int width = 512;
        int th=0, tw=0;
        int shape[3] = {batch, height, width};
        int minmn = height > width ? width : height;

        double* host_A = (double*)malloc(sizeof(double) * height * width);

        string matrix_path1 = "./data/robust_test/" + matrixes[iter1] + ".txt";

        // read in host A
        FILE* A_fp = fopen(matrix_path1.data(), "r");
        if(A_fp==NULL){
            printf("open file falied\n");
            return ;
        }

        for(int i=0; i < height*width; i++){
            fscanf(A_fp, "%lf", &host_A[i]);
        }

        fclose(A_fp);
        double iter2 = 5;
        for(int i=0;i<20; i++){
            double *dev_A;
            cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
            for(int i=0; i<batch; i++){
                cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
            }

            double *dev_U, *dev_V, *dev_diag;
            cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
            cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
            cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

            double test_result[4] = {10.0, 1.0, 1.0, iter2+i}; // 0:tag, 1:time
            cusolver_svd(dev_A, shape, dev_diag, dev_U, dev_V, test_result);

            double resi = get_residual(host_A, dev_U, dev_diag, dev_V, width);
            printf("max_sweep:%d, resi:%e\n", (int)iter2+i, resi);
            
            cudaFree(dev_A);
            cudaFree(dev_U);
            cudaFree(dev_V);
            cudaFree(dev_diag);
            cudaDeviceReset();                
        }

        free(host_A);
    }
}



// general test, input matrix and tailoring strategy, do SVD and check the result (compare to cusolver)
void test0(){
    FILE* input_fp = fopen("./io_files/in.txt", "r");
    if(input_fp == NULL){
        printf("open in.txt failed\n");
        return;
    }
    
    int batch;
    int height;
    int width;
    int th=0, tw=0;
    fscanf(input_fp, "%d %d %d %d %d", &batch, &height, &width, &th, &tw);
    fclose(input_fp);
    int shape[3] = {batch, height, width};
    int minmn = height > width ? width : height;

    printf("matrix:%d×%d×%d, tile:%d×%d\n", batch, height, width, th, tw);

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

    double *dev_A;
    cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
    for(int i=0; i<batch; i++){
        cudaMemcpy(dev_A + height*width*i, host_A, sizeof(double) * height * width, cudaMemcpyHostToDevice);
    }

    double *dev_U, *dev_V, *dev_diag;
    cudaMalloc((void **)&dev_diag, sizeof(double) * minmn * batch);
    cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
    cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);

    double test_result[4] = {0, 1.0, 1.0, 1.0}; // 0:tag, 1:time
    test_result[0] = 2.0;

    // our svd
    if(height <= 48 && width <= 48){
        svd_samll_matrix(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
    }
    else{
       svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, th, tw, test_result);
    }
    
    vector<double> re0(minmn * batch);
    vector<double> re1(height * height * batch);
    vector<double> re2(width * width * batch);
    cudaMemcpy(re0.data(), dev_diag, sizeof(double) * minmn * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(re1.data(), dev_U, sizeof(double) * height * height * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(re2.data(), dev_V, sizeof(double) * width * width * batch, cudaMemcpyDeviceToHost);

    for(int i=0; i<batch; i++){
        sort(re0.begin() + i*minmn, re0.begin() + (i+1)*minmn, greater<double>());
    }
    print_matrix(re0.data(), batch, minmn, "./io_files/out.txt");
    printf("svd time:%lf\n", test_result[1]);
    printf("singular value have been writed into ./io_files/out.txt\n");

    cudaMemset(dev_V, 0, sizeof(double) * width * width * batch);
	cudaMemset(dev_U, 0, sizeof(double) * height * height * batch);
    cudaMemset(dev_diag, 0, sizeof(double) * minmn * batch);


    // cusolver svd
    if(height <= 32 && width <= 32){
        cusolver_svd_batched(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
    }
    else{
       cusolver_svd(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
    }
    
    vector<double> re3(minmn * batch);
    vector<double> re4(height * height * batch);
    vector<double> re5(width * width * batch);
    cudaMemcpy(re3.data(), dev_diag, sizeof(double) * minmn * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(re4.data(), dev_U, sizeof(double) * height * height * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(re5.data(), dev_V, sizeof(double) * width * width * batch, cudaMemcpyDeviceToHost);

    print_matrix(re3.data(), batch, minmn, "./io_files/out_lib.txt");
    printf("cuslover svd time:%lf\n", test_result[2]);
    printf("singular value have been writed into ./io_files/out_lib.txt\n");

    free(host_A);
    cudaFree(dev_A);
    cudaFree(dev_U);
    cudaFree(dev_V);
    cudaFree(dev_diag);
    cudaDeviceReset();
}

// generate matrix set
void before_test(){
    int h_array[26] = {8, 8, 16, 16, 24, 24, 32, 32, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024, 1024, 32, 32, 16, 16};
    int w_array[26] = {8, 32, 16, 32, 24, 32, 8, 16, 24, 32, 64, 1024, 128, 1024, 256, 1024, 512, 1024, 64, 128, 512, 1024, 256, 512, 256, 512};

    int sq_array[8] = {100, 500, 1000, 2000, 4000, 5000, 8000, 10000};

    for(int i=0; i<26; i++){
        printf("generate matrix %d × %d             \r", h_array[i], w_array[i]);
        fflush(stdout);
        generate_matrix(h_array[i], w_array[i]);
    }

    for(int j=0; j<8; j++){
        printf("generate matrix %d × %d             \r", sq_array[j], sq_array[j]);
        fflush(stdout);
        generate_matrix(sq_array[j], sq_array[j]);
    }
    printf("\n over \n");
}

void before_test2(){
    generate_matrix_plus(512, 512, 11);
    printf("\n over \n");
}

void before_test3(){
    initCUDA();
    printf("over \n");
}

int main(int argc, char* argv[]){
    if(argc == 1){
        printf("================== test0 ==================\n");
        test0();
    }
    else if(argc == 2){
        int i;
        sscanf(argv[1], "%d", &i);
        if(i == 1){
            printf("================== test1 ==================\n");
            test1();
        }
        if(i == 2){
            printf("================== test2 ==================\n");
            test2();
            system("./test 2 1");
            system("./test 2 2");
            system("./test 2 3");
            system("./test 2 4");
        }
        if(i == 3){
            printf("================== test3 ==================\n");
            test3();
        }
        if(i == 4){
            printf("================== test4 ==================\n");
            test4();
        }
        if(i == 5){
            printf("================== test5 ==================\n");
            test5();
        }
        if(i == 6){
            printf("================== test6 ==================\n");
            test6();
        }
        if(i == 7){
            printf("================== test7 ==================\n");
            test7();
        }
        if(i == 8){
            printf("================== test8 ==================\n");
            test8();
        }
        if(i == 9){
            printf("================== test9 ==================\n");
            test9();
        }
        if(i == 10){
            printf("================== test10 ==================\n");
            test10();
        }
        if(i == 11){
            printf("================== test11 ==================\n");
            test11();
        }
        if(i == 14){
            printf("================== test14 ==================\n");
            test14();
        }
        if(i == 15){
            printf("================== test15 ==================\n");
            // test15();
            test15();
        }
        if(i == 16){
            printf("================== test16 ==================\n");
            test16();
        }
        if(i == 17){
            printf("================== test17 ==================\n");
            test17();
        }
        if(i == 22){
            printf("================== test22 ==================\n");
            test22();
        }


        if(i == 99){
            printf("================== generate matrix ==================\n");
            before_test();
        }
        if(i == 98){
            printf("================== generate matrix plus==================\n");
            before_test2();
        }
        if(i == 97){
            printf("================== device propertites ==================\n");
            before_test3();
        }
    }
    else if(argc == 3){
        int i, j;
        sscanf(argv[1], "%d", &i);
        sscanf(argv[2], "%d", &j);
        // printf("i:%d,j:%d\n", i, j);
        if(i == 2){
            test2(j);
        }
        if(i == 20){
            printf("================== test20 ==================\n");
            test20(j);
        }
        if(i == 21){
            printf("================== test21 ==================\n");
            test21(j);
        }

    }
    else if(argc == 4){
        int i, j, k;
        sscanf(argv[1], "%d", &i);
        sscanf(argv[2], "%d", &j);
        sscanf(argv[3], "%d", &k);
        // printf("i:%d,j:%d\n", i, j);
        if(i == 12){
            printf("intput: batch %d,height and width %d\n", j, k);
            test12(j, k);
        }
        if(i == 13){
            printf("intput: batch %d,height and width %d\n", j, k);
            test13(j, k);
        }
        if(i == 18){
            printf("intput: batch %d,height and width %d\n", j, k);
            test18(j, k);
        }
    }
    else if(argc == 5){
        int i, j, m, n;
        sscanf(argv[1], "%d", &i);
        sscanf(argv[2], "%d", &j);
        sscanf(argv[3], "%d", &m);
        sscanf(argv[4], "%d", &n);
        if(i == 19){
            printf("input batch:%d size:%dx%d, th:%d\n", j, m, m, n);
            test19(j, m, n);
        }
    }
    return 0;
}