#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "matrix_generate.hpp"
#include "large_matrix_svd.cu"
#include "small_matrix_svd.cu"
#include "cusolver_svd.cu"

using namespace std;

void test1();
void test2(int iter_in);
void test3();
void test4();
void test5();
void test6();
void test8();
void test9();
void test10();

// TODO(over)
void test1(){
    int h_array[2] = {32, 16};
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
    printf("size:     32×512  32×256  16×512  16×256\n");
    printf("SVD A_ij: %.3lf   %.3lf   %.3lf   %.3lf\n", time_result[0], time_result[1], time_result[2], time_result[3]);
    printf("SVD B_ij: %.3lf   %.3lf   %.3lf   %.3lf\n", time_result[4], time_result[5], time_result[6], time_result[7]);
    return;
}

// TODO(over)
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
    printf("= th     %d    %d    %d    %d    %d\n", th_array[0], th_array[1], th_array[2], th_array[3], th_array[3], th_array[4]);
    for(int i=0; i<5; i++){
        printf("= tw %d : %.2lf  %.2lf  %.2lf  %.2lf  %.2lf\n", tw_array[i], time_result[i*4 + 0], time_result[i*4 + 1], time_result[i*4 + 2], time_result[i*4 + 3], time_result[i*4 + 4]);
    }

    return;
}

// W-cycle SVD for improvement over cuSOLVER (small matrix)
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

// Comparison with cuSOLVER using different batch sizes
void test6(){
    vector<int> shape_array1 = {64, 128, 256, 512, 1024};
    vector<int> shape_array2 = {1024};
    vector<int> batch_array = {10, 100, 500};

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

// test auto tuning stratgy's improve
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
            svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);
            temp[1] = test_result[1];

            test_result[0] = 2.0;
            svd_large_matrix(dev_A, shape, dev_diag, dev_U, dev_V, 0, 0, test_result);
            temp[2] = test_result[1];

            
            printf("matrix:%d×%d×%d, over no tailoring:\n", batch, height, width);
            printf("tailoring on expertise speedup: %lf / %lf = %lf\n", temp[0], temp[1], temp[0]/temp[1]);
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

// TODO(over)
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

// florida matrix set test, over cusolver
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

// general test, input matrix and tailoring strategy, do SVD and check the result (compare to cusolver)
// 
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

        if(i == 99){
            printf("================== generate matrix ==================\n");
            before_test();
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
    }
    return 0;
}