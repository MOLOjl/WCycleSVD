#include <cuda.h>

#define AA_MAP(batch, pair, slice, y, x) ((batch)* k * k * sliceNum * p + (pair) * k * k * sliceNum + (slice) * k * k + (y) * k + (x))

__global__ void myevd_fused_batched(double* dev_AiAj, double* dev_AiAi, double *dev_AjAj, double *dev_jointG, int* dev_roundRobin, int p, int k, int sliceNum) {
    __shared__ double shared_G[32][32];
    __shared__ int shared_roundRobin[31][32];
    __shared__ int step;
    __shared__ double shared_V[32][32];
    __shared__ double shared_operators[2][32];
    __shared__ int shared_pairs[2][32];
    // shared_G[threadIdx.x][threadIdx.y] = dev_jointG[blockIdx.y * 2 * k * 2 * k*p + blockIdx.x * 2 * k * 2 * k + threadIdx.y * 2 * k + threadIdx.x];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    double inline_sum = 0;
    for (int slc=0; slc<sliceNum; ++slc)
    {
        if (ty < k) {
            if (tx < k) {
                inline_sum += dev_AiAi[AA_MAP(blockIdx.y, blockIdx.x, slc, ty, tx)];
            } else {
                inline_sum += dev_AiAj[AA_MAP(blockIdx.y, blockIdx.x, slc, ty, tx-k)];
            }
        } else {
            if ( tx < k ) {
                inline_sum += dev_AiAj[AA_MAP(blockIdx.y, blockIdx.x, slc, tx, ty-k)];
            } else {
                inline_sum += dev_AjAj[AA_MAP(blockIdx.y, blockIdx.x, slc, ty-k, tx-k)];
            }
        }
    }
    shared_G[tx][ty] = inline_sum;

    if (threadIdx.y < (2 * k - 1)) {
        shared_roundRobin[threadIdx.y][threadIdx.x] = dev_roundRobin[threadIdx.y * 2 * k + threadIdx.x];
    }
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        step = 0;
    }
    if (threadIdx.y == threadIdx.x) {
        shared_V[threadIdx.y][threadIdx.x] = 1;
    }
    else {
        shared_V[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();
    int index1 = 0, index2 = 0;
    double vi, temp;
    while (step < (2 * k - 1)) {
        if (threadIdx.y == 0) {
            if (threadIdx.x < k) {
                index1 = shared_roundRobin[step][threadIdx.x];
                index2 = shared_roundRobin[step][2 * k - 1 - threadIdx.x];
                shared_pairs[0][index1] = index1;
                shared_pairs[1][index1] = index2;
                shared_pairs[0][index2] = index1;
                shared_pairs[1][index2] = index2;

                if (shared_G[index1][index2] != 0) {
                    double tao = (shared_G[index1][index1] - shared_G[index2][index2]) / (2 * shared_G[index1][index2]);
                    double signTao;
                    if (tao > 0) signTao = 1;
                    if (tao == 0) signTao = 0;
                    if (tao < 0) signTao = -1;
                    double tan = signTao / ((fabs(tao) + sqrt(1 + tao * tao)));
                    double cos = 1 / sqrt(1 + tan * tan);
                    double sin = tan * cos;
                    shared_operators[0][index1] = cos;
                    shared_operators[1][index1] = sin;
                    shared_operators[0][index2] = -sin;
                    shared_operators[1][index2] = cos;
                }
                else {
                    shared_operators[0][index1] = 1;
                    shared_operators[1][index1] = 0;
                    shared_operators[0][index2] = 0;
                    shared_operators[1][index2] = 1;
                }
            }

        }

        __syncthreads();

        temp = shared_operators[0][threadIdx.y] * (shared_G[shared_pairs[0][threadIdx.y]][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
            shared_G[shared_pairs[0][threadIdx.y]][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x]) +
            shared_operators[1][threadIdx.y] * (shared_G[shared_pairs[1][threadIdx.y]][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
                shared_G[shared_pairs[1][threadIdx.y]][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x]);
        vi = shared_V[threadIdx.y][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
            shared_V[threadIdx.y][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x];

        __syncthreads();
        shared_G[threadIdx.y][threadIdx.x] = temp;
        shared_V[threadIdx.y][threadIdx.x] = vi;

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            step++;
        }
        __syncthreads();
    }

    dev_jointG[blockIdx.y * 2 * k * 2 * k*p + blockIdx.x * 2 * k * 2 * k + threadIdx.y * 2 * k + threadIdx.x] = shared_V[threadIdx.x][threadIdx.y];
}


__global__ void myevd_fused_batched_4(double* dev_AiAj, double* dev_AiAi, double *dev_AjAj, double *dev_jointG, int *dev_roundRobin, int p, int k, int sliceNum)
{
    __shared__ double shared_G[8][8];
    __shared__ int shared_roundRobin[7][8];
    __shared__ int step;
    __shared__ double shared_V[8][8];
    __shared__ double shared_operators[2][8];
    __shared__ int shared_pairs[2][8];

    // shared_G[threadIdx.x][threadIdx.y] = dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + threadIdx.y * 2 * k + threadIdx.x];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    double inline_sum = 0;
    for (int slc=0; slc<sliceNum; ++slc)
    {
        if (tx < k) {
            if (ty < k) {
                inline_sum += dev_AiAi[AA_MAP(blockIdx.y, blockIdx.x, slc, ty, tx)];
            } else {
                inline_sum += dev_AiAj[AA_MAP(blockIdx.y, blockIdx.x, slc, ty-k, tx)];
            }
        } else {
            if ( ty < k ) {
                inline_sum += dev_AiAj[AA_MAP(blockIdx.y, blockIdx.x, slc, tx-k, ty)];
            } else {
                inline_sum += dev_AjAj[AA_MAP(blockIdx.y, blockIdx.x, slc, ty-k, tx-k)];
            }
        }
    }
    shared_G[tx][ty] = inline_sum;

    if (threadIdx.y < (2 * k - 1))
    {
        shared_roundRobin[threadIdx.y][threadIdx.x] = dev_roundRobin[threadIdx.y * 2 * k + threadIdx.x];
    }
    if (threadIdx.y == 0 && threadIdx.x == 0)
    {
        step = 0;
    }
    if (threadIdx.y == threadIdx.x)
    {
        shared_V[threadIdx.y][threadIdx.x] = 1;
    }
    else
    {
        shared_V[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();
    int index1 = 0, index2 = 0;
    double vi, temp;
    while (step < (2 * k - 1))
    {
        if (threadIdx.y == 0)
        {
            if (threadIdx.x < k)
            {
                index1 = shared_roundRobin[step][threadIdx.x];
                index2 = shared_roundRobin[step][2 * k - 1 - threadIdx.x];
                shared_pairs[0][index1] = index1;
                shared_pairs[1][index1] = index2;
                shared_pairs[0][index2] = index1;
                shared_pairs[1][index2] = index2;

                if (shared_G[index1][index2] != 0)
                {
                    double tao = (shared_G[index1][index1] - shared_G[index2][index2]) / (2 * shared_G[index1][index2]);
                    double signTao;
                    if (tao > 0)
                        signTao = 1;
                    if (tao == 0)
                        signTao = 1;	// shouldn't eq to 0
                    if (tao < 0)
                        signTao = -1;
                    double tan = signTao / ((fabs(tao) + sqrt(1 + tao * tao)));
                    double cos = 1 / sqrt(1 + tan * tan);
                    double sin = tan * cos;
                    shared_operators[0][index1] = cos;
                    shared_operators[1][index1] = sin;
                    shared_operators[0][index2] = -sin;
                    shared_operators[1][index2] = cos;
                }
                else
                {
                    shared_operators[0][index1] = 1;
                    shared_operators[1][index1] = 0;
                    shared_operators[0][index2] = 0;
                    shared_operators[1][index2] = 1;
                }
            }
        }

        __syncthreads();

        temp = shared_operators[0][threadIdx.y] * (shared_G[shared_pairs[0][threadIdx.y]][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
                                                   shared_G[shared_pairs[0][threadIdx.y]][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x]) +
               shared_operators[1][threadIdx.y] * (shared_G[shared_pairs[1][threadIdx.y]][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
                                                   shared_G[shared_pairs[1][threadIdx.y]][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x]);
        vi = shared_V[threadIdx.y][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
             shared_V[threadIdx.y][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x];

        __syncthreads();
        shared_G[threadIdx.y][threadIdx.x] = temp;
        shared_V[threadIdx.y][threadIdx.x] = vi;

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            step++;
        }
        __syncthreads();
    }

    dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + threadIdx.y * 2 * k + threadIdx.x] = shared_V[threadIdx.x][threadIdx.y];
}


__global__ void myevd_fused_batched_8(double* dev_AiAj, double* dev_AiAi, double *dev_AjAj, double *dev_jointG, int *dev_roundRobin, int p, int k, int sliceNum)
{
    __shared__ double shared_G[16][16];
    __shared__ int shared_roundRobin[15][16];
    __shared__ int step;
    __shared__ double shared_V[16][16];
    __shared__ double shared_operators[2][16];
    __shared__ int shared_pairs[2][16];
    // shared_G[threadIdx.x][threadIdx.y] = dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + threadIdx.y * 2 * k + threadIdx.x];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    double inline_sum = 0;
    for (int slc=0; slc<sliceNum; ++slc)
    {
        if (tx < k) {
            if (ty < k) {
                inline_sum += dev_AiAi[AA_MAP(blockIdx.y, blockIdx.x, slc, ty, tx)];
            } else {
                inline_sum += dev_AiAj[AA_MAP(blockIdx.y, blockIdx.x, slc, ty-k, tx)];
            }
        } else {
            if ( ty < k ) {
                inline_sum += dev_AiAj[AA_MAP(blockIdx.y, blockIdx.x, slc, tx-k, ty)];
            } else {
                inline_sum += dev_AjAj[AA_MAP(blockIdx.y, blockIdx.x, slc, ty-k, tx-k)];
            }
        }
    }
    shared_G[tx][ty] = inline_sum;

    if (threadIdx.y < (2 * k - 1))
    {
        shared_roundRobin[threadIdx.y][threadIdx.x] = dev_roundRobin[threadIdx.y * 2 * k + threadIdx.x];
    }
    if (threadIdx.y == 0 && threadIdx.x == 0)
    {
        step = 0;
    }
    if (threadIdx.y == threadIdx.x)
    {
        shared_V[threadIdx.y][threadIdx.x] = 1;
    }
    else
    {
        shared_V[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();
    int index1 = 0, index2 = 0;
    double vi, temp;
    while (step < (2 * k - 1))
    {
        if (threadIdx.y == 0)
        {
            if (threadIdx.x < k)
            {
                index1 = shared_roundRobin[step][threadIdx.x];
                index2 = shared_roundRobin[step][2 * k - 1 - threadIdx.x];
                shared_pairs[0][index1] = index1;
                shared_pairs[1][index1] = index2;
                shared_pairs[0][index2] = index1;
                shared_pairs[1][index2] = index2;

                if (shared_G[index1][index2] != 0)
                {
                    double tao = (shared_G[index1][index1] - shared_G[index2][index2]) / (2 * shared_G[index1][index2]);
                    double signTao;
                    if (tao > 0)
                        signTao = 1;
                    if (tao == 0)
                        signTao = 1;	// shouldn't eq to 0
                    if (tao < 0)
                        signTao = -1;
                    double tan = signTao / ((fabs(tao) + sqrt(1 + tao * tao)));
                    double cos = 1 / sqrt(1 + tan * tan);
                    double sin = tan * cos;
                    shared_operators[0][index1] = cos;
                    shared_operators[1][index1] = sin;
                    shared_operators[0][index2] = -sin;
                    shared_operators[1][index2] = cos;
                }
                else
                {
                    shared_operators[0][index1] = 1;
                    shared_operators[1][index1] = 0;
                    shared_operators[0][index2] = 0;
                    shared_operators[1][index2] = 1;
                }
            }
        }

        __syncthreads();

        temp = shared_operators[0][threadIdx.y] * (shared_G[shared_pairs[0][threadIdx.y]][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
                                                   shared_G[shared_pairs[0][threadIdx.y]][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x]) +
               shared_operators[1][threadIdx.y] * (shared_G[shared_pairs[1][threadIdx.y]][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
                                                   shared_G[shared_pairs[1][threadIdx.y]][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x]);
        vi = shared_V[threadIdx.y][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
             shared_V[threadIdx.y][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x];

        __syncthreads();
        shared_G[threadIdx.y][threadIdx.x] = temp;
        shared_V[threadIdx.y][threadIdx.x] = vi;

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            step++;
        }
        __syncthreads();
    }

    dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + threadIdx.y * 2 * k + threadIdx.x] = shared_V[threadIdx.x][threadIdx.y];
}


__global__ void myevd_fused_batched_16(double* dev_AiAj, double* dev_AiAi, double *dev_AjAj, double *dev_jointG, int *dev_roundRobin, int p, int k, int sliceNum)
{
    __shared__ double shared_G[32][32];
    __shared__ int shared_roundRobin[31][32];
    __shared__ double shared_V[32][32];
    __shared__ double shared_operators[2][32];
    __shared__ int shared_pairs[2][32];
    __shared__ int step;
    
    // shared_G[threadIdx.y][threadIdx.x] = dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + threadIdx.x * 2 * k + threadIdx.y];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    double inline_sum=0;
    
    if (ty < k) {
        if (tx < k) {
            for (int slc=0; slc<sliceNum; ++slc)
            {
                // if(AA_MAP(blockIdx.y, blockIdx.x, slc, ty, tx) >= gridDim.y*p*k*k*sliceNum) printf("(%d,%d)FUCK ", ty, tx);
                inline_sum += dev_AiAi[AA_MAP(blockIdx.y, blockIdx.x, slc, ty, tx)];
            }
        } else {
            for (int slc=0; slc<sliceNum; ++slc)
            {
                // if(AA_MAP(blockIdx.y, blockIdx.x, slc, ty, tx-k) >= gridDim.y*p*k*k*sliceNum) printf("(%d,%d)FUCK ", ty, tx);
                inline_sum += dev_AiAj[AA_MAP(blockIdx.y, blockIdx.x, slc, ty, tx-k)];
            }
        }
    } else {
        if ( tx < k ) {
            for (int slc=0; slc<sliceNum; ++slc)
            {
                // if(AA_MAP(blockIdx.y, blockIdx.x, slc, tx, ty-k) >= gridDim.y*p*k*k*sliceNum) printf("(%d,%d)FUCK ", ty, tx);
                inline_sum += dev_AiAj[AA_MAP(blockIdx.y, blockIdx.x, slc, tx, ty-k)];
            }
        } else {
            for (int slc=0; slc<sliceNum; ++slc)
            {
                // if(AA_MAP(blockIdx.y, blockIdx.x, slc, ty-k, tx-k) >= gridDim.y*p*k*k*sliceNum) printf("(%d,%d)FUCK ", ty, tx);
                inline_sum += dev_AjAj[AA_MAP(blockIdx.y, blockIdx.x, slc, ty-k, tx-k)];
            }
        }
    }

    // if (ty < k && tx < k)
    // {
    //     for (int slc=0; slc<sliceNum; ++slc)
    //     {
    //         inline_sum[0] += dev_AiAi[AA_MAP(blockIdx.y, blockIdx.x, slc, ty, tx)];
    //         inline_sum[1] += dev_AiAj[AA_MAP(blockIdx.y, blockIdx.x, slc, ty, tx)];
    //         inline_sum[2] += dev_AiAj[AA_MAP(blockIdx.y, blockIdx.x, slc, tx, ty)];
    //         inline_sum[3] += dev_AjAj[AA_MAP(blockIdx.y, blockIdx.x, slc, ty, tx)];
    //     }
    //     shared_G[tx][ty] = inline_sum[0];
    //     shared_G[tx+k][ty] = inline_sum[1];
    //     shared_G[tx][ty+k] = inline_sum[1];
    //     shared_G[tx+k][ty+k] = inline_sum[1];
    // }
    shared_G[tx][ty] = inline_sum;
    // __syncthreads();
    // if (tx == 0 && ty == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    // {
    //     for (int i=0; i<2*k; i++) {
    //         for (int j=0; j<2*k; j++)
    //             printf("%.2f ", shared_G[j][i]);
    //         printf("\n");
    //     }
        
    //     printf("\n");

    //     for (int i=0; i<2*k; i++) {
    //         for (int j=0; j<2*k; j++)
    //             printf("%.2f ", dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + i * 2 * k + j]);
    //         printf("\n");
    //     }

    //     printf("\n");
    //     double sum = 0;
    //     for (int i=0; i<sliceNum; i++) sum += dev_AiAi[AA_MAP(0, 0, i, 0, 0)];
    //     printf("[0,0]=%f\n", sum); 
    //     printf("\n\n\n\n");
    // }
    // __syncthreads();

    if (threadIdx.y < (2 * k - 1))
    {
        shared_roundRobin[threadIdx.y][threadIdx.x] = dev_roundRobin[threadIdx.y * 2 * k + threadIdx.x];
    }
    if (threadIdx.y == 0 && threadIdx.x == 0)
    {
        step = 0;
    }

    shared_V[threadIdx.y][threadIdx.x] = (threadIdx.y==threadIdx.x);

    __syncthreads();
    int index1 = 0, index2 = 0;
    double vi, temp;
    while (step < (2 * k - 1))
    {
        if (threadIdx.y == 0)
        {
            if (threadIdx.x < k)	// (0~16)
            {
                // 取数+存数
                index1 = shared_roundRobin[step][threadIdx.x];
                index2 = shared_roundRobin[step][2 * k - 1 - threadIdx.x];
                shared_pairs[0][index1] = index1;
                shared_pairs[1][index1] = index2;
                shared_pairs[0][index2] = index1;
                shared_pairs[1][index2] = index2;

                if (shared_G[index1][index2] != 0)
                {
                    double tao = (shared_G[index1][index1] - shared_G[index2][index2]) / (2 * shared_G[index1][index2]);
                    double signTao;
                    if (tao > 0)
                        signTao = 1;
                    if (tao == 0)
                        signTao = 1;	// shouldn't eq to 0
                    if (tao < 0)
                        signTao = -1;
                    double tan = signTao / ((fabs(tao) + sqrt(1 + tao * tao)));
                    double cos = 1 / sqrt(1 + tan * tan);
                    double sin = tan * cos;
                    shared_operators[0][index1] = cos;
                    shared_operators[1][index1] = sin;
                    shared_operators[0][index2] = -sin;
                    shared_operators[1][index2] = cos;
                }
                else
                {
                    shared_operators[0][index1] = 1;
                    shared_operators[1][index1] = 0;
                    shared_operators[0][index2] = 0;
                    shared_operators[1][index2] = 1;
                }
            }
        }

        __syncthreads();

        temp = shared_operators[0][threadIdx.y] * (shared_G[shared_pairs[0][threadIdx.y]][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
                                                   shared_G[shared_pairs[0][threadIdx.y]][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x]) +
               shared_operators[1][threadIdx.y] * (shared_G[shared_pairs[1][threadIdx.y]][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
                                                   shared_G[shared_pairs[1][threadIdx.y]][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x]);
        vi = shared_V[threadIdx.y][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
             shared_V[threadIdx.y][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x];

        __syncthreads();

        shared_G[threadIdx.y][threadIdx.x] = temp;
        shared_V[threadIdx.y][threadIdx.x] = vi;

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            step++;
        }
        __syncthreads();
    }

    dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + threadIdx.y * 2 * k + threadIdx.x] = shared_V[threadIdx.x][threadIdx.y];
    __syncthreads();
}


//dim3 dimGrid9(p, batch, 1);
//dim3 dimBlock9(k, k, 1);
__global__ void myevd_fused_batched_24(double* dev_AiAj, double* dev_AiAi, double *dev_AjAj, double *dev_jointG, int *dev_roundRobin, int p, int k, int sliceNum)
{
    __shared__ double shared_G[48][48];
    __shared__ int shared_roundRobin[47][48];
    __shared__ int step;
    __shared__ double shared_V[48][48];
    __shared__ double shared_operators[2][48];
    __shared__ int shared_pairs[2][48];

    //2k*2k
    // shared_G[threadIdx.x][threadIdx.y] = dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + threadIdx.y * 2 * k + threadIdx.x];
    // shared_G[threadIdx.x + k][threadIdx.y] = dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + threadIdx.y * 2 * k + threadIdx.x + k];
    // shared_G[threadIdx.x][threadIdx.y + k] = dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + (threadIdx.y + k) * 2 * k + threadIdx.x];
    // shared_G[threadIdx.x + k][threadIdx.y + k] = dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + (threadIdx.y + k) * 2 * k + threadIdx.x + k];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    double inline_sum[4]{};
    for (int slc=0; slc<sliceNum; ++slc)
    {
        inline_sum[0] += dev_AiAi[AA_MAP(blockIdx.y, blockIdx.x, slc, ty, tx)];
        inline_sum[1] += dev_AiAj[AA_MAP(blockIdx.y, blockIdx.x, slc, ty, tx)];
        inline_sum[2] += dev_AiAj[AA_MAP(blockIdx.y, blockIdx.x, slc, tx, ty)];
        inline_sum[3] += dev_AjAj[AA_MAP(blockIdx.y, blockIdx.x, slc, ty, tx)];
    }
    shared_G[tx][ty] = inline_sum[0];
    shared_G[tx+k][ty] = inline_sum[1];
    shared_G[tx][ty+k] = inline_sum[2];
    shared_G[tx+k][ty+k] = inline_sum[3];

    //(2 * k - 1)* 2k
    shared_roundRobin[threadIdx.x][threadIdx.y] = dev_roundRobin[threadIdx.x * 2 * k + threadIdx.y];
    shared_roundRobin[threadIdx.x][threadIdx.y + k] = dev_roundRobin[threadIdx.x * 2 * k + threadIdx.y + k];
    if (threadIdx.x < (k - 1))
    {
        shared_roundRobin[threadIdx.x + k][threadIdx.y] = dev_roundRobin[(threadIdx.x + k) * 2 * k + threadIdx.y];
        shared_roundRobin[threadIdx.x + k][threadIdx.y + k] = dev_roundRobin[(threadIdx.x + k) * 2 * k + threadIdx.y + k];
    }

    if (threadIdx.y == 0 && threadIdx.x == 0)
    {
        step = 0;
    }
    shared_V[threadIdx.y][threadIdx.x] = 0;
    shared_V[threadIdx.y + k][threadIdx.x] = 0;
    shared_V[threadIdx.y][threadIdx.x + k] = 0;
    shared_V[threadIdx.y + k][threadIdx.x + k] = 0;
    __syncthreads();
    if (threadIdx.y == threadIdx.x)
    {
        shared_V[threadIdx.y][threadIdx.x] = 1;
        shared_V[threadIdx.y + k][threadIdx.x + k] = 1;
    }
    else {
        shared_V[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();
    int index1 = 0, index2 = 0;
    double vi0, temp0, vi1, temp1, vi2, temp2, vi3, temp3;
    while (step < (2 * k - 1))
    {
        if (threadIdx.y == 0)
        {
            index1 = shared_roundRobin[step][threadIdx.x];
            index2 = shared_roundRobin[step][2 * k - 1 - threadIdx.x];
            shared_pairs[0][index1] = index1;
            shared_pairs[1][index1] = index2;
            shared_pairs[0][index2] = index1;
            shared_pairs[1][index2] = index2;

            if (shared_G[index1][index2] != 0)
            {
                double tao = (shared_G[index1][index1] - shared_G[index2][index2]) / (2 * shared_G[index1][index2]);
                double signTao;
                if (tao > 0)
                    signTao = 1;
                if (tao == 0)
                    signTao = 1;	// shouldn't eq to 0
                if (tao < 0)
                    signTao = -1;
                double tan = signTao / ((fabs(tao) + sqrt(1 + tao * tao)));
                double cos = 1 / sqrt(1 + tan * tan);
                double sin = tan * cos;
                shared_operators[0][index1] = cos;
                shared_operators[1][index1] = sin;
                shared_operators[0][index2] = -sin;
                shared_operators[1][index2] = cos;
            }
            else
            {
                shared_operators[0][index1] = 1;
                shared_operators[1][index1] = 0;
                shared_operators[0][index2] = 0;
                shared_operators[1][index2] = 1;
            }
        }
        __syncthreads();
        //x,y
        temp0 = shared_operators[0][threadIdx.y] * (shared_G[shared_pairs[0][threadIdx.y]][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
                                                    shared_G[shared_pairs[0][threadIdx.y]][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x]) +
                shared_operators[1][threadIdx.y] * (shared_G[shared_pairs[1][threadIdx.y]][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
                                                    shared_G[shared_pairs[1][threadIdx.y]][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x]);
        vi0 = shared_V[threadIdx.y][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
              shared_V[threadIdx.y][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x];
        //x+k,y
        temp1 = shared_operators[0][threadIdx.y] * (shared_G[shared_pairs[0][threadIdx.y]][shared_pairs[0][threadIdx.x + k]] * shared_operators[0][threadIdx.x + k] +
                                                    shared_G[shared_pairs[0][threadIdx.y]][shared_pairs[1][threadIdx.x + k]] * shared_operators[1][threadIdx.x + k]) +
                shared_operators[1][threadIdx.y] * (shared_G[shared_pairs[1][threadIdx.y]][shared_pairs[0][threadIdx.x + k]] * shared_operators[0][threadIdx.x + k] +
                                                    shared_G[shared_pairs[1][threadIdx.y]][shared_pairs[1][threadIdx.x + k]] * shared_operators[1][threadIdx.x + k]);
        vi1 = shared_V[threadIdx.y][shared_pairs[0][threadIdx.x + k]] * shared_operators[0][threadIdx.x + k] +
              shared_V[threadIdx.y][shared_pairs[1][threadIdx.x + k]] * shared_operators[1][threadIdx.x + k];
        //x,y+k
        temp2 = shared_operators[0][threadIdx.y + k] * (shared_G[shared_pairs[0][threadIdx.y + k]][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
                                                        shared_G[shared_pairs[0][threadIdx.y + k]][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x]) +
                shared_operators[1][threadIdx.y + k] * (shared_G[shared_pairs[1][threadIdx.y + k]][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
                                                        shared_G[shared_pairs[1][threadIdx.y + k]][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x]);
        vi2 = shared_V[threadIdx.y + k][shared_pairs[0][threadIdx.x]] * shared_operators[0][threadIdx.x] +
              shared_V[threadIdx.y + k][shared_pairs[1][threadIdx.x]] * shared_operators[1][threadIdx.x];
        //x+k,y+k
        temp3 = shared_operators[0][threadIdx.y + k] * (shared_G[shared_pairs[0][threadIdx.y + k]][shared_pairs[0][threadIdx.x + k]] * shared_operators[0][threadIdx.x + k] +
                                                        shared_G[shared_pairs[0][threadIdx.y + k]][shared_pairs[1][threadIdx.x + k]] * shared_operators[1][threadIdx.x + k]) +
                shared_operators[1][threadIdx.y + k] * (shared_G[shared_pairs[1][threadIdx.y + k]][shared_pairs[0][threadIdx.x + k]] * shared_operators[0][threadIdx.x + k] +
                                                        shared_G[shared_pairs[1][threadIdx.y + k]][shared_pairs[1][threadIdx.x + k]] * shared_operators[1][threadIdx.x + k]);
        vi3 = shared_V[threadIdx.y + k][shared_pairs[0][threadIdx.x + k]] * shared_operators[0][threadIdx.x + k] +
              shared_V[threadIdx.y + k][shared_pairs[1][threadIdx.x + k]] * shared_operators[1][threadIdx.x + k];
        __syncthreads();
        shared_G[threadIdx.y][threadIdx.x] = temp0;
        shared_V[threadIdx.y][threadIdx.x] = vi0;
        shared_G[threadIdx.y][threadIdx.x + k] = temp1;
        shared_V[threadIdx.y][threadIdx.x + k] = vi1;
        shared_G[threadIdx.y + k][threadIdx.x] = temp2;
        shared_V[threadIdx.y + k][threadIdx.x] = vi2;
        shared_G[threadIdx.y + k][threadIdx.x + k] = temp3;
        shared_V[threadIdx.y + k][threadIdx.x + k] = vi3;

        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            step++;
        }
        __syncthreads();
    }

    dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + threadIdx.y * 2 * k + threadIdx.x] = shared_V[threadIdx.x][threadIdx.y];
    dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + threadIdx.y * 2 * k + threadIdx.x + k] = shared_V[threadIdx.x + k][threadIdx.y];
    dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + (threadIdx.y + k) * 2 * k + threadIdx.x] = shared_V[threadIdx.x][threadIdx.y + k];
    dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + (threadIdx.y + k) * 2 * k + threadIdx.x + k] = shared_V[threadIdx.x + k][threadIdx.y + k];
}

#undef AA_MAP
