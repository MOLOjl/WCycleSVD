#include <stdlib.h>
#include <time.h>
#include <chrono>

#include <algorithm>
#include <cusolverDn.h>
#include <vector>

#include "small_matrix_svd_kernels.cu"
#include "result_print.cu"
#include "utils.cu"

void svd_small_matrix(double* dev_A, int* shape, double* dev_diag, double* dev_U, double* dev_V, double* test_tag=nullptr)
{
    int batch = shape[0];
    int height = shape[1];
    int width = shape[2];
	int *dev_roundRobin;

    cudaMemset(dev_V, 0, sizeof(double) * width * width * batch);
	cudaMemset(dev_U, 0, sizeof(double) * height * height * batch);

    clock_t start_cpu, stop_cpu;
    double our_time = 0;
    if (height >= width)
    {
        if (width % 2 == 0)
        {
            cudaMalloc((void **)&dev_roundRobin, sizeof(int) * (width - 1) * width);
            cudaDeviceSynchronize();
            start_cpu = clock();    //&2.3
            
            dim3 dimGrid0(1, 1, 1);
            // dim3 dimBlock0(width / 2, width / 2, 1);
            // generate_roundRobin_64<<<dimGrid0, dimBlock0>>>(dev_roundRobin, width / 2); //&1.1
            dim3 dimBlock0(32, 32, 1);  // 一个block最多开1024个线程
            generate_roundRobin_128<<<dimGrid0, dimBlock0>>>(dev_roundRobin, width); //&1.1      

            cudaDeviceSynchronize();
            // printf("%d * %d * %d, hi\n", batch, height, width);
            if(test_tag != nullptr && test_tag[0] == 12.0){
                printf("hello tag 12\n");
                small_svd_even_column_1_warp <<<batch, 16>>> (dev_A, height, width, dev_U, dev_V, dev_diag, dev_roundRobin);  //&1.2
            }
            else{
                small_svd_even_column<<<batch, 16 * width/2>>> (dev_A, height, width, dev_U, dev_V, dev_diag, dev_roundRobin);  //&1.2
            }
            
            cudaDeviceSynchronize();
            stop_cpu = clock(); //&2.3
            our_time = (double)(stop_cpu - start_cpu) / CLOCKS_PER_SEC;
        }
        else
        {
            cudaMalloc((void **)&dev_roundRobin, sizeof(int) * width * (width + 1));
            cudaDeviceSynchronize();
            start_cpu = clock();
            dim3 dimGrid0(1, 1, 1);
            dim3 dimBlock0((width + 1) / 2, (width + 1) / 2, 1);
            generate_roundRobin_64<<<dimGrid0, dimBlock0>>>(dev_roundRobin, (width + 1) / 2);   //&1.1
            cudaDeviceSynchronize();
            small_svd_odd_column<<<batch, 16 * (width + 1) / 2>>>(dev_A, height, width, dev_U, dev_V, dev_diag, dev_roundRobin);    //&1.2
            cudaDeviceSynchronize();
            stop_cpu = clock();
            our_time = (double)(stop_cpu - start_cpu) / CLOCKS_PER_SEC;
        }
    }
    else
    {
        if (height % 2 == 0)
        {
            cudaMalloc((void **)&dev_roundRobin, sizeof(int) * (height - 1) * height);
            cudaDeviceSynchronize();
            start_cpu = clock();
            dim3 dimGrid0(1, 1, 1);
            dim3 dimBlock0(height / 2, height / 2, 1);
            generate_roundRobin_64<<<dimGrid0, dimBlock0>>>(dev_roundRobin, height / 2);    //&1.1
            cudaDeviceSynchronize();
            small_svd_even_column_trans<<<batch, 16 * height / 2>>>(dev_A, height, width, dev_U, dev_V, dev_diag, dev_roundRobin);  //&1.2
            cudaDeviceSynchronize();
            stop_cpu = clock();
            our_time = (double)(stop_cpu - start_cpu) / CLOCKS_PER_SEC;
        }
        else
        {
            cudaMalloc((void **)&dev_roundRobin, sizeof(int) * height * (height + 1));
            cudaDeviceSynchronize();
            start_cpu = clock();
            dim3 dimGrid0(1, 1, 1);
            dim3 dimBlock0((height + 1) / 2, (height + 1) / 2, 1);
            generate_roundRobin_64<<<dimGrid0, dimBlock0>>>(dev_roundRobin, (height + 1) / 2);  //&1.1
            cudaDeviceSynchronize();
            small_svd_odd_column_trans<<<batch, 16 * (height + 1) / 2>>>(dev_A, height, width, dev_U, dev_V, dev_diag, dev_roundRobin); //&1.2
            cudaDeviceSynchronize();
            stop_cpu = clock();
            our_time = (double)(stop_cpu - start_cpu) / CLOCKS_PER_SEC;
        }
    }

    if(test_tag!=nullptr){
        test_tag[1] = our_time;
    }
    
    cudaFree(dev_roundRobin);    //>5
}
