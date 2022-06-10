#include <stdlib.h>
#include <time.h>
#include <chrono>

#include <algorithm>
#include <cusolverDn.h>
#include <vector>

#include "large_matrix_svd_kernels.cu"
#include "result_print.cu"
#include "utils.cu"


/**
@param dev_A A matrix, sizes batch * height * width, input
@param shape int array, {n, h, w}, input
*/
void svd_large_matrix(double* dev_A00, int* shape, double* dev_diag0, double* dev_U0, double* dev_V0, int th=0, int tw=0, double* test_tag=nullptr)
{
    // dealwith input parameters
    int batch = shape[0];
    int height0 = shape[1];
    int width0 = shape[2];

    double* dev_A0 = dev_A00;
    double* temp_A = nullptr;
    int tag = 0;
    // do trans
    int do_trans = 1;
    if(test_tag!=nullptr && (test_tag[0]==3.0 || test_tag[0] == 6.0))
        do_trans = 0;
    if(height0 < width0 && do_trans){
        if(width0>1024){
            printf("not support\n");
            return;
        }
        cudaMalloc((void **)&temp_A, sizeof(double) * height0 * width0 * batch);
        transform<<<batch, 1024>>>(dev_A00, temp_A, height0, width0);
        dev_A0 = temp_A;
        int t = height0;
        height0 = width0;
        width0 = t;

        double* temp_p = dev_V0;
        dev_V0 = dev_U0;
        dev_U0 = temp_p;

        tag = 1;
    }

    // decide tile shape
    if(tw == 0 || th==0){
        // auto tuning    
        double thres = 385000;
		int bn[4] = {24,16,8,4 };
		int t[4]={ 256,256,256,128 };
		int bm[6] = { 1024,512,256,128,64,32};
		int ii=0,jj=0;
		while(bm[jj]>height0) jj++;
		
		int p = (width0 - 1) / (2 * bn[ii]) + 1;
		int q = (height0 - 1) / bm[jj] + 1;
		int tlp=batch*p*q*t[ii];

		for (;(ii < 3 ||jj<5) && tlp > thres;) {
			if(ii < 3){
				ii++;
				p = (width0 - 1) / (2 * bn[ii]) + 1;
		        tlp=batch*p*q*t[ii];
			}
			if(jj<5&& tlp > thres){
				jj++;
		        q = (height0 - 1) / bm[jj] + 1;
		        tlp=batch*p*q*t[ii];
			}
		}
        tw=16*2; th=bm[jj];
        
        if(height0>=256){
           tw=32; th=128;
        }
        if(height0>=512){
            tw=32; th=256;
        }
    }
    else{
        if(tw!=2 && tw!=8 && tw!=16 && tw!=32 && tw!=48){
            if(test_tag[0] != 4.0){
                printf("invalid tw %d\n", tw);
                return;
            }
        }
        if(th%32!=0 || th>height0){
            if(test_tag[0] != 3.0 && test_tag[0] != 6.0){
                printf("invalid th %d\n", th);
                return;                
            }
        }
    }


    if(test_tag != nullptr){
        if(test_tag[0] == 0){
            // no tailoring
            tw = 48;
            int bm_iter = 0;
            int bm[6] = {1024,512,256,128,64,32};
            while(bm_iter < 6){
                th = bm[bm_iter];
                if(th <= height0)
                    break;
                bm_iter ++;
            }
        }
        if(test_tag[0] == 1.0){
            // exp
            tw=32; th=32;
        }
    }

    // printf("input matrix shape: %d × %d × %d, tile shape: %d × %d\n", batch, height0, width0, th, tw);

// prams
#pragma region
    int k = tw/2;
    int slice = th;
    /* p is the count of match-matrix A_ij, 
    e.g. a 16*16 matrix，k=4, 16*8 match-matrix A_ij's count is 2, i.e. p=2. */
    int p = (width0 - 1) / (2 * k) + 1; 
    // printf("pairs = %d\n", p);
    // each match-matrix A_ij is cut into slices at column wise, q is the count of these slices 
    int q = (height0 - 1) / slice + 1;
    int width = p * (2 * k);
    int height = q * slice;
    int sliceNum = q;
    
    double* dev_A;  // fixed A
    double* dev_V;
    double* dev_U;
	int* dev_roundRobin; 
    
    double* dev_jointG;
    double* dev_Aij;

    double* dev_AiAi;   
    double* dev_AiAj;
    double* dev_AjAj;
    int* dev_pairsOfEVD;
    unsigned* host_allpass;
    unsigned* host_pass;
    unsigned* dev_allpass;
    unsigned* dev_pass;
    double *value;
    unsigned int *arr1;
    unsigned int *arr2;
    unsigned int *pairs;
    double *dev_norm;
    unsigned int *dev_order;
    double* host_Fnorm; 
    double* dev_tempFnorm;
    double* dev_Fnorm;
    double* dev_diag;
#pragma endregion

// memory allocate
#pragma region
    cudaMalloc((void **)&dev_A, sizeof(double) * height * width * batch);
    cudaMalloc((void **)&dev_V, sizeof(double) * width * width * batch);
    dev_U = dev_U0;
    dev_diag = dev_diag0;

    // cudaMalloc((void **)&dev_U, sizeof(double) * height * height * batch);
    // cudaMalloc((void **)&dev_V0, sizeof(double) * width0 * width0 * batch);

    cudaMalloc((void **)&dev_roundRobin, sizeof(int) * (2 * k - 1) * 2 * k);

    cudaMalloc((void **)&dev_jointG, sizeof(double) * 2*k * 2*k * p*batch);
    cudaMalloc((void **)&dev_Aij, sizeof(double) * height * 2*k * p*batch);

    cudaMalloc((void **)&dev_AiAi, sizeof(double) * k * k * sliceNum * p * batch);
    cudaMalloc((void **)&dev_AiAj, sizeof(double) * k * k * sliceNum * p * batch);
    cudaMalloc((void **)&dev_AjAj, sizeof(double) * k * k * sliceNum * p * batch);
    cudaMalloc((void **)&dev_pairsOfEVD, sizeof(int) * 2 * p * batch);

    host_allpass = (unsigned *)malloc(sizeof(unsigned) * batch);
    host_pass = (unsigned *)malloc(sizeof(unsigned) * p * batch);
    cudaMalloc((void **)&dev_allpass, sizeof(unsigned) * batch);
    cudaMalloc((void **)&dev_pass, sizeof(unsigned) * p * batch);

    cudaMalloc((void **)&dev_norm, sizeof(double) * 2 * p * batch);
    cudaMalloc((void **)&dev_order, sizeof(unsigned int) * 2 * p * batch);
    host_Fnorm = (double *)malloc(sizeof(double) * batch);
    cudaMalloc((void **)&dev_tempFnorm, sizeof(double) * 2 * p * batch);
    cudaMalloc((void **)&dev_Fnorm, sizeof(double) * batch);

#pragma endregion

// preset before svd  
#pragma region

    cudaMemset(dev_V, 0, sizeof(double) * width * width * batch);   // 64×64×100 的矩阵
    cudaMemset(dev_U, 0, sizeof(double) * height0 * height0 * batch);
    cudaMemset(dev_V0, 0, sizeof(double) * width0 * width0 * batch);
    cudaMemset(dev_pairsOfEVD, 0, sizeof(int) * 2 * p * batch); 
    memset(host_pass, 0, sizeof(unsigned) * p * batch);
    cudaMemset(dev_pass, 0, sizeof(unsigned) * p * batch);

    cudaDeviceSynchronize();
    // clock_t start_cpu, stop_cpu;
    // start_cpu = clock();

    // fix A, width/k==0 && height/32==0
    dim3 dimGrid12(2 * p * height/32, batch, 1);
    if(height >= 32)
        generateDev_A<<<dimGrid12, 128>>>(dev_A0, dev_A, dev_V, height0, width0, height, width, p, k, height/32);
    else{
        cudaMemcpy(dev_A, dev_A0, sizeof(double)* height * width * batch, cudaMemcpyDeviceToDevice);
        init_dev_V<<<batch, 256>>>(dev_V, width);
    }
    cudaDeviceSynchronize();

    // compute frobenius norm of A, do this as slice=32, q=height/slice
    if(height >= 32){
        computeFnorm1<<<2 * p * batch, 128>>>(dev_A, dev_tempFnorm, p, height/32, height, width, k);
    }
    else{
        computeFnorm1<<<2 * p * batch, 128>>>(dev_A, dev_tempFnorm, p, 1, height, width, k);   
    }
    cudaDeviceSynchronize();
    computeFnorm2<<<batch, 32>>>(dev_tempFnorm, dev_Fnorm, p);  //&1.3

    dim3 dimGrid0(1, 1, 1);
    dim3 dimBlock0(32, 32, 1);
    // generate_roundRobin_64<<<dimGrid0, dimBlock0>>>(dev_roundRobin, k); //&1.1
    generate_roundRobin_128<<<dimGrid0, dimBlock0>>>(dev_roundRobin, 2*k);

    getRankNewNew<<<1, 1024>>>(2 * p);  //&1.3
    cudaDeviceSynchronize();

    // compute as q=height/32
    parallel_ordering_choice(p, height/32, dev_order, dev_A, height, width, dev_norm, batch, k);    //&1.3
    memset(host_allpass, 0, sizeof(unsigned) * batch);
    int sweep = 0;

#pragma endregion

// cublas init
#pragma region

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    const double alpha = 1.0;
    const double beta = 0.0;

    double **d_Aij_array = nullptr;
    double **d_G_array = nullptr;
    cudaMalloc((void **)&d_Aij_array, sizeof(double*) * p * batch);
    cudaMalloc((void **)&d_G_array, sizeof(double*) * p * batch);
    set_d_AG_array<<<1, 1>>>(dev_Aij, dev_jointG, d_Aij_array, d_Aij_array, d_G_array, p*batch, height, 2*k);

    // double* re = (double*)malloc(sizeof(double)*height*width);
    // cudaMemcpy(re, dev_A, sizeof(double)*height*width, cudaMemcpyDeviceToHost);
    // print_matrix(re, height, width, "result1.txt");
#pragma endregion
    
    // clock_t start, end;
    // start = clock();

    double start, end, gemm_reduce=0;
    start = wtime();

    int maxsweep = 20;
    while (/*!ifallpass(host_allpass, batch, p) &&*/ sweep<maxsweep)//占时少
    {   
        for (int i = 0; i < 2 * p - 1; i++)
        // for (int i = 0; i < 1; i++)
        {
#pragma region  // original svd

            if(test_tag != nullptr && test_tag[0] == 3.0){
                dim3 dimGrid_fG(p, batch);
                match_Aij<<<dimGrid_fG, 256>>>(dev_A, height, width, i, dev_order, dev_pairsOfEVD, dev_Aij, p, k);
                cudaDeviceSynchronize();

                cublasDgemmBatched(handle, transa, transb, 2*k, 2*k, height, &alpha, d_Aij_array, height, d_Aij_array, height, &beta, d_G_array, 2*k, p*batch);
                cudaDeviceSynchronize();

                converge_verify<<<dimGrid_fG, 256>>>(dev_jointG, 2*k, dev_Fnorm, dev_pass, p, k, i);
                cudaDeviceSynchronize();

                mysvd_batched_even1<<<p*batch, 16 * k>>>(dev_Aij, 2*k, 2*k, dev_Aij, dev_V, dev_roundRobin);
                cudaDeviceSynchronize();
                
                dim3 dimGrid12(p, batch);
			    update_AV<<<dimGrid12, 64>>>(dev_A, dev_V, dev_Aij, dev_pairsOfEVD, p, height, width, k);
            }
            else if(test_tag != nullptr && test_tag[0] == 6.0){
                dim3 dimGrid_fG(p, batch);
                match_Aij<<<dimGrid_fG, 256>>>(dev_A, height, width, i, dev_order, dev_pairsOfEVD, dev_Aij, p, k);
                cudaDeviceSynchronize();

                cublasDgemmBatched(handle, transa, transb, 2*k, 2*k, height, &alpha, d_Aij_array, height, d_Aij_array, height, &beta, d_G_array, 2*k, p*batch);
                cudaDeviceSynchronize();

                converge_verify<<<dimGrid_fG, 256>>>(dev_jointG, 2*k, dev_Fnorm, dev_pass, p, k, i);
                cudaDeviceSynchronize();

                mysvd_batched_even1<<<p*batch, 16 * k>>>(dev_jointG, 2*k, 2*k, dev_jointG, dev_V, dev_roundRobin);
                cudaDeviceSynchronize();
                
                dim3 dimGrid12(p, batch);
			    update_AV<<<dimGrid12, 64>>>(dev_A, dev_V, dev_jointG, dev_pairsOfEVD, p, height, width, k);
            }
            else{
                if(k <= 24){
                    // normally evd B_ij
                    dim3 dimGrid77(sliceNum, p, batch); // 2×2×100个block，每个block 256线程
                #ifndef USE_TC
                    generate_jointG00<<<dimGrid77, 256>>>(dev_A, height, width, dev_order, dev_pass, p, q, dev_pairsOfEVD, dev_AiAi, dev_AiAj, dev_AjAj, i, k, slice, sliceNum);    //&1.3
                    cudaDeviceSynchronize();
                #else
                    if (k == 4)
                    {
                        // double *h_ans, *h_res;
                        // double *check_AiAi, *check_AiAj, *check_AjAj;
                        // cudaMalloc((void **)&check_AiAi, sizeof(double) * k * k * sliceNum * p * batch);
                        // cudaMalloc((void **)&check_AiAj, sizeof(double) * k * k * sliceNum * p * batch);
                        // cudaMalloc((void **)&check_AjAj, sizeof(double) * k * k * sliceNum * p * batch);
                        // h_ans = new double[k * k * sliceNum * p * batch];
                        // h_res = new double[k * k * sliceNum * p * batch];
                        generate_jointG00_tc<4><<<dimGrid77, dim3(32)>>>(dev_A, height, width, dev_order, dev_pass, p, q, dev_pairsOfEVD, dev_AiAi, dev_AiAj, dev_AjAj, i, slice, sliceNum);    //&1.3
                        
                        // cudaMemcpy(h_ans, dev_AiAj, sizeof(double) * k * k * sliceNum * p * batch, cudaMemcpyDeviceToHost );
                        // cudaMemcpy(h_res, check_AiAj, sizeof(double) * k * k * sliceNum * p * batch, cudaMemcpyDeviceToHost );

                        // // check AiAi
                        // for ( int i=0; i < k * k * sliceNum * p * batch; ++i )
                        // {
                        //     if (fabs(h_ans[i] - h_res[i]) > 1e-5)
                        //     {
                        //         int offset = i / 576 * 576;
                        //         int intra = i % 576;
                        //         int intra_x = i % 24;
                        //         int intra_y = i / 24;
                        //         printf("false [%d]: %f(x) <- %f\n", i, h_res[i], h_ans[i]);
                        //         printf("sym [%d] = %f\n", intra_x*24+intra_y, h_res[offset+intra_x*24+intra_y]);
                        //         printf("next_row [%d]: %f <- %f\n", i+23, h_res[i+23], h_ans[i+23]);
                        //         exit(0);
                        //     }
                        // }

                        // cudaMemcpy(h_ans, dev_AiAi, sizeof(double) * k * k * sliceNum * p * batch, cudaMemcpyDeviceToHost );
                        // cudaMemcpy(h_res, check_AiAi, sizeof(double) * k * k * sliceNum * p * batch, cudaMemcpyDeviceToHost );
                        // // check AiAj
                        // for ( int i=0; i < k * k * sliceNum * p * batch; ++i )
                        // {
                        //     if (fabs(h_ans[i] - h_res[i]) > 1e-5)
                        //     {
                        //         int offset = i / 576 * 576;
                        //         int intra = i % 576;
                        //         int intra_x = i % 24;
                        //         int intra_y = i / 24;
                        //         printf("false [%d]: %f(x) <- %f\n", i, h_res[i], h_ans[i]);
                        //         printf("sym [%d] = %f\n", intra_x*24+intra_y, h_res[offset+intra_x*24+intra_y]);
                        //         printf("next_row [%d]: %f <- %f\n", i+23, h_res[i+23], h_ans[i+23]);
                        //         exit(0);
                        //     }
                        // }

                        // cudaMemcpy(h_ans, dev_AjAj, sizeof(double) * k * k * sliceNum * p * batch, cudaMemcpyDeviceToHost );
                        // cudaMemcpy(h_res, check_AjAj, sizeof(double) * k * k * sliceNum * p * batch, cudaMemcpyDeviceToHost );
                        // // check AjAj
                        // for ( int i=0; i < k * k * sliceNum * p * batch; ++i )
                        // {
                        //     if (fabs(h_ans[i] - h_res[i] )> 1e-5)
                        //     {
                        //         int offset = i / 576 * 576;
                        //         int intra = i % 576;
                        //         int intra_x = i % 24;
                        //         int intra_y = i / 24;
                        //         printf("false [%d]: %f(x) <- %f\n", i, h_res[i], h_ans[i]);
                        //         printf("sym [%d] = %f\n", intra_x*24+intra_y, h_res[offset+intra_x*24+intra_y]);
                        //         printf("next_row [%d]: %f <- %f\n", i+23, h_res[i+23], h_ans[i+23]);
                        //         exit(0);
                        //     }
                        // }

                        // cudaFree(check_AiAi);
                        // cudaFree(check_AiAj);
                        // cudaFree(check_AjAj);
                        // delete [] h_ans, h_res;
                    } else if ( k == 8 ) {
                        generate_jointG00_tc<8><<<dimGrid77, dim3(32,k/8,k/8)>>>(dev_A, height, width, dev_order, dev_pass, p, q, dev_pairsOfEVD, dev_AiAi, dev_AiAj, dev_AjAj, i, slice, sliceNum);    //&1.3
                    } else if ( k == 16 ) {
                        // if(sweep==0 && i==0) printf("hi\n");
                        generate_jointG00_tc<16><<<dimGrid77, dim3(32,k/8,k/8)>>>(dev_A, height, width, dev_order, dev_pass, p, q, dev_pairsOfEVD, dev_AiAi, dev_AiAj, dev_AjAj, i, slice, sliceNum);    //&1.3
                    } else { // k == 24
                        generate_jointG00_tc<24><<<dimGrid77, dim3(32,k/8,k/8)>>>(dev_A, height, width, dev_order, dev_pass, p, q, dev_pairsOfEVD, dev_AiAi, dev_AiAj, dev_AjAj, i, slice, sliceNum);    //&1.3
                    }
                    cudaDeviceSynchronize();
                #endif
                    

                    dim3 dimGrid7(p, batch, 1);
                    generate_jointG21<<<dimGrid7, 256>>>(dev_jointG, dev_AiAi, dev_AiAj, dev_AjAj, dev_Fnorm, dev_pass, p, k, sliceNum);    //&1.3
                    cudaDeviceSynchronize();

                    gemm_reduce += EVD_(dev_jointG, dev_AiAj, dev_AiAi, dev_AjAj, dev_A, dev_V, dev_pairsOfEVD, p, q, height, width, dev_roundRobin, batch, k, slice, sliceNum, sweep); //&1.3
                    // cudaDeviceSynchronize();                       
                }
                else{
                    dim3 dimGrid_fG(p, batch);
                    match_Aij<<<dimGrid_fG, 256>>>(dev_A, height, width, i, dev_order, dev_pairsOfEVD, dev_Aij, p, k);
                    cudaDeviceSynchronize();

                    cublasDgemmBatched(handle, transa, transb, 2*k, 2*k, height, &alpha, d_Aij_array, height, d_Aij_array, height, &beta, d_G_array, 2*k, p*batch);
                    cudaDeviceSynchronize();

                    converge_verify<<<dimGrid_fG, 256>>>(dev_jointG, 2*k, dev_Fnorm, dev_pass, p, k, i);
                    cudaDeviceSynchronize();
                    if(sweep == maxsweep-1)
                        gemm_reduce = EVD_(dev_AiAj, dev_AiAi, dev_AjAj, dev_jointG, dev_A, dev_V, dev_pairsOfEVD, p, q, height, width, dev_roundRobin, batch, k, slice, sliceNum, 99);
                    else
                        gemm_reduce = EVD_(dev_AiAj, dev_AiAi, dev_AjAj, dev_jointG, dev_A, dev_V, dev_pairsOfEVD, p, q, height, width, dev_roundRobin, batch, k, slice, sliceNum, sweep);
                }
            }

#pragma endregion
        }

        parallel_ordering_choice(p, height/32, dev_order, dev_A, height, width, dev_norm, batch, k);    // may changed some match orders
        judgeFunc<<<batch, 1024>>>(dev_allpass, dev_pass, p);   // concentrate each block's result(converged or not)
        cudaDeviceSynchronize();
        cudaMemcpy(host_allpass, dev_allpass, sizeof(unsigned) * batch, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // printf("sweep %d :completed.\n", sweep);
        sweep++;
    }
    cudaDeviceSynchronize();

    dim3 dimGrid10(2 * p, batch, 1);
    dim3 dimBlock10(32, k, 1);
    getUDV<<<dimGrid10, dimBlock10>>>(dev_A, dev_U, dev_V, dev_V0, height, width, height0, width0, p, height/32, dev_diag, width0, k);  //&1.3
    cudaDeviceSynchronize();

    // end = clock();
    end = wtime();
    double our_time = (double)(end - start); /// CLOCKS_PER_SEC;
    // printf("sweep:%d, time:%lf\n",sweep, our_time);
    if(test_tag != nullptr){
        if(test_tag[0] == 4.0){
            test_tag[3] = sweep;
        }
        test_tag[1] = our_time;
        test_tag[2] = gemm_reduce;
    }

// free
#pragma region
    if(temp_A!=nullptr)
        cudaFree(temp_A);
    cudaFree(dev_A);    //>2
    cudaFree(dev_V);    //>3
    cudaFree(dev_roundRobin);   //>6
    cudaFree(dev_jointG);   //>7
    cudaFree(dev_AiAi); //>8
    cudaFree(dev_AiAj); //>9
    cudaFree(dev_AjAj); //>10
    cudaFree(dev_pairsOfEVD);   //>11
    free(host_allpass); //>12
    free(host_pass);    //>13
    cudaFree(dev_allpass);  //>14
    cudaFree(dev_pass); //>15
    cudaFree(dev_norm); //>20
    cudaFree(dev_order);    //>21
    free(host_Fnorm);   //>22
    cudaFree(dev_tempFnorm);    //>23
    cudaFree(dev_Fnorm);    //>24

#pragma endregion
}