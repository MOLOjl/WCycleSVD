#ifndef CUSOLVER_SVD
#define CUSOLVER_SVD

#include <stdlib.h>
#include <string>
#include <vector>
#include <chrono>
#include <time.h>
#include <cusolverDn.h>
#include "result_print.cu"


using namespace std;

/**
 * @brief serial cuslover svd for any shape matrix
 * 
 * @param dev_A A matrix, sizes batch * height * width, input
 * @param shape int array, {n, h, w}, input
 * @param dev_diag sigma matirx, output
 * @param dev_U left singlar matrix, output
 * @param dev_V right singlar matrix, output
 */
void cusolver_svd(double* dev_A, int* shape, double* dev_diag, double* dev_U, double* dev_V, double* test_tag=nullptr){
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    gesvdjInfo_t gesvdj_params = NULL;

    const int batch = shape[0];
    const int m = shape[1];
    const int n = shape[2];

    // printf("input matrix shape: %d × %d × %d\n", batch, m, n);
    const int lda = m;
    const int ldu = m;
    const int ldv = n;
    const int minmn = min(m,n);

    int *d_info = NULL;  /* error info */
    int lwork = 0;       /* size of workspace */
    double *d_work = NULL; /* devie workspace for gesvdj */
    int info = 0;        /* host copy of error info */
    
    /* configuration of gesvdj  */
    const double tol = 1e-10;
    int max_sweeps = 20;
    if(test_tag != nullptr && test_tag[0] == 10.0){
        // specify maxsweep
        max_sweeps = (int)test_tag[3];
    }

    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
    const int econ = 0 ; /* econ = 1 for economy size */

    double residual = 0;
    int executed_sweeps = 0;

    cusolverDnCreate(&cusolverH);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cusolverDnSetStream(cusolverH, stream);

    cusolverDnCreateGesvdjInfo(&gesvdj_params);
    // cusolverDnXgesvdjSetTolerance(gesvdj_params, tol);
    cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps);
    cudaMalloc((void**)&d_info, sizeof(int));

    cusolverDnDgesvdj_bufferSize(cusolverH, jobz, econ, m, n, dev_A, lda, dev_diag, dev_U, ldu, dev_V, ldv, &lwork, gesvdj_params);

    cudaMalloc((void**)&d_work , sizeof(double)*lwork);

    clock_t start, end;
    start = clock();

    for(int i=0; i<batch; i++){
        cusolverDnDgesvdj(cusolverH, jobz, econ, m, n, dev_A + lda*n*i, lda, dev_diag + minmn*i, dev_U + ldu*m*i, ldu, dev_V + ldv*n*i, ldv, d_work, lwork, d_info, gesvdj_params);
        cudaDeviceSynchronize();
    }

    end = clock();
    double lib_time = (double)(end - start) / CLOCKS_PER_SEC;
    if(test_tag != nullptr){
        test_tag[2] = lib_time;
    }

    if (d_info)  cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (stream) cudaStreamDestroy(stream);
    if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);
}

/**
 * @brief cusolver Dn(dense) D(double) ge(general) svd j(jacobi) Batched, matrix shape shall lessequal than 32
 * 
 * @param dev_A A matrix, sizes batch * height * width, input
 * @param shape int array, {n, h, w}, input
 * @param dev_diag sigma matirx, output
 * @param dev_U left singlar matrix, output
 * @param dev_V right singlar matrix, output
 */
void cusolver_svd_batched(double* dev_A, int* shape, double* dev_diag, double* dev_U, double* dev_V, double* test_tag=nullptr){
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    gesvdjInfo_t gesvdj_params = NULL;

    const int batch = shape[0];
    const int m = shape[1];   /* 1 <= m <= 32 */
    const int n = shape[2];   /* 1 <= n <= 32 */

    if(m>32 || n>32)
    {
        printf("matrix too big\n");
        return;
    }

    const int lda = m; /* lda >= m */
    const int ldu = m; /* ldu >= m */
    const int ldv = n; /* ldv >= n */
    const int minmn = (m < n) ? m : n; /* min(m,n) */

    std::vector<int> info(batch, 0);             /* info = [info0 ; info1] */

    int *d_info = nullptr; /* batchSize */
    int lwork = 0;            /* size of workspace */
    double *d_work = nullptr; /* device workspace for getrf */

    const double tol = 1.e-7;
    const int max_sweeps = 15;
    const int sort_svd = 0;                                  /* don't sort singular values */
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; /* compute singular vectors */


    /* step 1: create cusolver handle, bind a stream */
    cusolverDnCreate(&cusolverH);

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cusolverDnSetStream(cusolverH, stream);

    /* step 2: configuration of gesvdj */
    cusolverDnCreateGesvdjInfo(&gesvdj_params);

    /* default value of tolerance is machine zero */
    cusolverDnXgesvdjSetTolerance(gesvdj_params, tol);

    /* default value of max. sweeps is 100 */
    cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps);

    /* disable sorting */
    cusolverDnXgesvdjSetSortEig(gesvdj_params, sort_svd);

    /* step 3: copy A to device */
    cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int) * info.size());


    /* step 4: query working space of gesvdjBatched */
    cusolverDnDgesvdjBatched_bufferSize(cusolverH, jobz, m, n, dev_A, lda, dev_diag, dev_U, ldu, dev_V, ldv, &lwork, gesvdj_params, batch);
    cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork);
    
    clock_t start, end;
    start = clock();

    /* step 5: compute singular values of A */
    cusolverDnDgesvdjBatched(cusolverH, jobz, m, n, dev_A, lda, dev_diag, dev_U, ldu, dev_V, ldv, d_work, lwork, d_info, gesvdj_params, batch);
    cudaStreamSynchronize(stream);

    end = clock();
    double lib_time = (double)(end - start) / CLOCKS_PER_SEC;
    if(test_tag != nullptr){
        test_tag[2] = lib_time;
    }

    cudaMemcpyAsync(info.data(), d_info, sizeof(int) * info.size(), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    /* Step 6: show singular values and singular vectors */
    // print_matrix(S.data(), 32, 32);

    /* free resources */
    cudaFree(d_info);
    cudaFree(d_work);

    cusolverDnDestroyGesvdjInfo(gesvdj_params);
    cusolverDnDestroy(cusolverH);
    cudaStreamDestroy(stream);
    // cudaDeviceReset();
}


#endif