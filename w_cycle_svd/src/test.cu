#include <stdio.h>
#include <stdlib.h>
#include "large_matrix_svd.cu"
#include "small_matrix_svd.cu"
#include "cusolver_svd.cu"

using namespace std;

// test auto tuning stratgy's improve
void test1(){
    
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
            string matrix_path1 = "../data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";
            string matrix_path2 = "./data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";   // in case
            // read in host A
            FILE* A_fp = fopen(matrix_path1.data(), "r");

            if(A_fp==NULL){
                A_fp = fopen(matrix_path2.data(), "r");
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
            printf("tailoring on experance speedup: %lf / %lf = %lf\n", temp[0], temp[1], temp[0]/temp[1]);
            printf("tailoring on Auto-tuning speedup: %lf / %lf = %lf\n", temp[0], temp[2], temp[0]/temp[2]);
            printf("==================================================================\n");

            free(host_A);
            cudaFree(dev_A);
            cudaFree(dev_U);
            cudaFree(dev_V);
            cudaFree(dev_diag);
            cudaDeviceReset();
            // cusolver_svd(dev_A, shape, dev_diag, dev_U, dev_V, test_result);
            
            // vector<double> re0(minmn);
            // vector<double> re1(height * height * batch);
            // vector<double> re2(width * width * batch);

            // cudaMemcpy(re0.data(), &dev_diag[minmn*7], sizeof(double) * minmn, cudaMemcpyDeviceToHost);
            // cudaMemcpy(re1.data(), dev_U, sizeof(double) * height * height * batch, cudaMemcpyDeviceToHost);
            // cudaMemcpy(re2.data(), dev_V, sizeof(double) * width * width * batch, cudaMemcpyDeviceToHost);

            // sort(re0.begin(), re0.end(), greater<double>());
            // print_matrix(re0.data(), 32, 32, "result.txt");
            // break;
        }
        // break;
    }
    return;

}

// Comparison with cuSOLVER using different batch sizes
void test2(){
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

                    string matrix_path1 = "../data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";
                    string matrix_path2 = "./data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";   // in case
                    // read in host A
                    FILE* A_fp = fopen(matrix_path1.data(), "r");

                    if(A_fp==NULL){
                        A_fp = fopen(matrix_path2.data(), "r");
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
                    
                    // vector<double> re0(minmn);
                    // vector<double> re1(height * height * batch);
                    // vector<double> re2(width * width * batch);
                    // cudaMemcpy(re0.data(), &dev_diag[minmn*7], sizeof(double) * minmn, cudaMemcpyDeviceToHost);
                    // cudaMemcpy(re1.data(), dev_U, sizeof(double) * height * height * batch, cudaMemcpyDeviceToHost);
                    // cudaMemcpy(re2.data(), dev_V, sizeof(double) * width * width * batch, cudaMemcpyDeviceToHost);
                    // sort(re0.begin(), re0.end(), greater<double>());
                    // print_matrix(re0.data(), 32, 32, "result.txt");
                    // break;
                
                }           
                
                if(iter1 != 4)
                {
                    int batch = batch_array[iter3];
                    int width = shape_array1[iter1];
                    int height = shape_array2[iter2];
                    int shape[3] = {batch, height, width};
                    int minmn = height > width ? width : height;
                    
                    double* host_A = (double*)malloc(sizeof(double) * height * width);

                    string matrix_path1 = "../data/generated_matrixes/A_h" + to_string(width) + "_w" + to_string(height)+ ".txt";
                    string matrix_path2 = "./data/generated_matrixes/A_h" + to_string(width) + "_w" + to_string(height)+ ".txt";   // in case

                    // read in host A
                    FILE* A_fp = fopen(matrix_path1.data(), "r");

                    if(A_fp==NULL){
                        A_fp = fopen(matrix_path2.data(), "r");
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
                    int height = shape_array1[iter1];
                    int width = shape_array1[iter1];
                    int shape[3] = {batch, height, width};
                    int minmn = height > width ? width : height;
                    
                    double* host_A = (double*)malloc(sizeof(double) * height * width);

                    string matrix_path1 = "../data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";
                    string matrix_path2 = "./data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";   // in case
                    // read in host A
                    FILE* A_fp = fopen(matrix_path1.data(), "r");

                    if(A_fp==NULL){
                        A_fp = fopen(matrix_path2.data(), "r");
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

// W-cycle SVD for improvement over cuSOLVER (small matrix)
void test3(){
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

                    string matrix_path1 = "../data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";
                    string matrix_path2 = "./data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";   // in case
                    // read in host A
                    FILE* A_fp = fopen(matrix_path1.data(), "r");

                    if(A_fp==NULL){
                        A_fp = fopen(matrix_path2.data(), "r");
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

                    string matrix_path1 = "../data/generated_matrixes/A_h" + to_string(width) + "_w" + to_string(height)+ ".txt";
                    string matrix_path2 = "./data/generated_matrixes/A_h" + to_string(width) + "_w" + to_string(height)+ ".txt";   // in case

                    // read in host A
                    FILE* A_fp = fopen(matrix_path1.data(), "r");

                    if(A_fp==NULL){
                        A_fp = fopen(matrix_path2.data(), "r");
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

                    string matrix_path1 = "../data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";
                    string matrix_path2 = "./data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";   // in case
                    // read in host A
                    FILE* A_fp = fopen(matrix_path1.data(), "r");

                    if(A_fp==NULL){
                        A_fp = fopen(matrix_path2.data(), "r");
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
void test4(){
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

            string matrix_path1 = "../data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";
            string matrix_path2 = "./data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";   // in case
            // read in host A
            FILE* A_fp = fopen(matrix_path1.data(), "r");

            if(A_fp==NULL){
                A_fp = fopen(matrix_path2.data(), "r");
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

// florida matrix set test, over cusolver
#define RANK_COUNT 5
#define MAX_BATCH 301
void test5(){
    int matrix_shapes[5] = {32, 64, 128, 256, 512};

    for(int i=1; i<=5; i++){
        string folder_path = "../data/UF_matrixset/" + to_string(i) + "/";
        string folder_path1 = "./data/UF_matrixset/" + to_string(i) + "/";

        double* host_A = (double*)malloc(sizeof(double)*MAX_BATCH * matrix_shapes[i-1]*matrix_shapes[i-1]);
        int probe = 0;
        int j = 0; // matrix count
        while(true){
            j ++;       
            string matrix_path = folder_path + to_string(j) + ".txt";
            string matrix_path1 = folder_path1 + to_string(j) + ".txt";

            FILE* matrix_fp = fopen(matrix_path.data(), "r");
            if(matrix_fp == NULL){
                matrix_fp = fopen(matrix_path1.data(), "r");
                if(matrix_fp == NULL){
                    break;
                }
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
        input_fp = fopen("../io_files/in.txt", "r");
        if(input_fp == NULL){
            printf("open in.txt failed\n");
            return;
        }
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

    string matrix_path1 = "../data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";
    string matrix_path2 = "./data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";   // in case
    string matrix_path3 = "../data/generated_matrixes/A_h" + to_string(width) + "_w" + to_string(height)+ ".txt";
    string matrix_path4 = "./data/generated_matrixes/A_h" + to_string(width) + "_w" + to_string(height)+ ".txt";

    // read in host A
    FILE* A_fp = fopen(matrix_path1.data(), "r");
    if(A_fp==NULL){
        A_fp = fopen(matrix_path2.data(), "r");
        if(A_fp==NULL){
            A_fp = fopen(matrix_path3.data(), "r");
            if(A_fp==NULL){
                A_fp = fopen(matrix_path4.data(), "r");
                if(A_fp==NULL){
                    printf("open file falied\n");
                    return ;
                }
            }
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
    print_matrix(re0.data(), batch, minmn, "../io_files/out.txt");
    printf("svd time:%lf\n", test_result[1]);
    printf("singular value have been writed into ../io_files/out.txt\n");

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

    print_matrix(re3.data(), batch, minmn, "../io_files/out_lib.txt");
    printf("cuslover svd time:%lf\n", test_result[2]);
    printf("singular value have been writed into ../io_files/out_lib.txt\n");

    free(host_A);
    cudaFree(dev_A);
    cudaFree(dev_U);
    cudaFree(dev_V);
    cudaFree(dev_diag);
    cudaDeviceReset();
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
        }
        if(i == 3){
            printf("================== test3 ==================\n");
            test3();
        }
        if(i == 4){
            printf("================== test4 ==================\n");
            test5();
        }
        if(i == 5){
            printf("================== test5 ==================\n");
            test5();
        }       
    }
    return 0;
}