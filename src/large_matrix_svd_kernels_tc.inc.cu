#include "tc_large_matrix_svd_kernels.cuh"
// tensor core solution works for k = 8,16,24
// <<<(sliceNum, p, batch), K/8,K/8,32>>>
template <int K>
__global__ void generate_jointG00_tc(double *dev_A, int height, int width, unsigned int *dev_order, unsigned int *dev_pass, int p, int q, int *dev_pairsOfEVD,
                                     double *dev_AiAi, double *dev_AiAj, double *dev_AjAj, int iterNum, int slice, int sliceNum)
{
    #define I_MAP(batch, y, x) ((batch)*height*width+(y)+(x)*height)
    #define O_MAP(batch, pair, slice, y, x) ((batch)* K * K * sliceNum * p + (pair) * K * K * sliceNum + (slice) * K * K + (y) * K + (x))
    
    using namespace nvcuda;

    __shared__ int index[2];
    // 必须按4double对齐
    __shared__ double sm_Ai[K][36]; //padded 
    __shared__ double sm_Aj[K][36];

    const int lane = threadIdx.x;  // 0~255
    const int wx = threadIdx.y;
    const int wy = threadIdx.z;

    int iter = slice / 32;	// 1

    if (lane == 0 && wx == 0 && wy == 0)
    {
        index[0] = dev_order[(2 * p - 1) - p_ab[0][blockIdx.y][iterNum] + blockIdx.z * 2 * p];
        index[1] = dev_order[(2 * p - 1) - p_ab[1][blockIdx.y][iterNum] + blockIdx.z * 2 * p];
        dev_pairsOfEVD[2 * (blockIdx.z * p + blockIdx.y)] = index[0];
        dev_pairsOfEVD[2 * (blockIdx.z * p + blockIdx.y) + 1] = index[1];
        
        // if(blockIdx.x==1 && blockIdx.y==1)
        // printf("index: %d - %d \n", index[0], index[1]);
    }

    __syncthreads();

    wmma::fragment<wmma::accumulator, 8, 8, 4, double> frag_As[3];

    #pragma unroll
    for ( int i=0; i<3; ++i) wmma::fill_fragment(frag_As[i], 0);

    for (int t = 0; t < iter; t++)
    {
        for ( int i=wx; i<8; i+=blockDim.y )
        {
            const int col = 8*wy+i;
            sm_Ai[col][lane] = dev_A[I_MAP(blockIdx.z, blockIdx.x*slice+t*32+lane, index[0]*K+col)];
        }

        for ( int i=wy; i<8; i+=blockDim.z )
        {
            const int col = 8*wx+i;
            sm_Aj[col][lane] = dev_A[I_MAP(blockIdx.z, blockIdx.x*slice+t*32+lane, index[1]*K+col)];
        }
        __syncthreads();

        // do GEMM
        do_wmma3_shared_tile<36,36>(sm_Ai, sm_Aj, 32, wy, wx, &frag_As[0]);
    
        __syncthreads();
    }

    wmma::store_matrix_sync(dev_AiAi+O_MAP(blockIdx.z, blockIdx.y, blockIdx.x, 8*wx, 8*wy), frag_As[0], K, wmma::mem_col_major);
    wmma::store_matrix_sync(dev_AjAj+O_MAP(blockIdx.z, blockIdx.y, blockIdx.x, 8*wx, 8*wy), frag_As[1], K, wmma::mem_col_major);
    wmma::store_matrix_sync(dev_AiAj+O_MAP(blockIdx.z, blockIdx.y, blockIdx.x, 8*wx, 8*wy), frag_As[2], K, wmma::mem_col_major);
    
    // store data

    #undef I_MAP
    #undef O_MAP
}

// specialization for solution works for k = 4
// <<<(sliceNum, p, batch), 32>>>
template <>
__global__ void generate_jointG00_tc<4>(double *dev_A, int height, int width, unsigned int *dev_order, unsigned int *dev_pass, int p, int q, int *dev_pairsOfEVD,
                                     double *dev_AiAi, double *dev_AiAj, double *dev_AjAj, int iterNum, int slice, int sliceNum)
{
    #define K 4
    #define I_MAP(batch, y, x) ((batch)*height*width+(y)+(x)*height)
    #define O_MAP(batch, pair, slice, y, x) ((batch)* K * K * sliceNum * p + (pair) * K * K * sliceNum + (slice) * K * K + (y) * K + (x))
    
    using namespace nvcuda;

    __shared__ int index[2];
    // 必须按4double对齐
    __shared__ double sm_Ai[8][36]; //padded 
    __shared__ double sm_Aj[8][36];

    const int lane = threadIdx.x;  // 0~255

    int iter = slice / 32;	// 1

    if (lane == 0)
    {
        index[0] = dev_order[(2 * p - 1) - p_ab[0][blockIdx.y][iterNum] + blockIdx.z * 2 * p];
        index[1] = dev_order[(2 * p - 1) - p_ab[1][blockIdx.y][iterNum] + blockIdx.z * 2 * p];
        dev_pairsOfEVD[2 * (blockIdx.z * p + blockIdx.y)] = index[0];
        dev_pairsOfEVD[2 * (blockIdx.z * p + blockIdx.y) + 1] = index[1];
        
        // if(blockIdx.x==1 && blockIdx.y==1)
        // printf("index: %d - %d \n", index[0], index[1]);
    }

    wmma::fragment<wmma::accumulator, 8, 8, 4, double> frag_As;

    #pragma unroll
    for ( int i=0; i<3; ++i) wmma::fill_fragment(frag_As, 0);

    for (int t = 0; t < iter; t++)
    {
        for ( int i=0; i<4; i+=blockDim.y )
        {
            sm_Ai[i][lane] = dev_A[I_MAP(blockIdx.z, blockIdx.x*slice+t*32+lane, index[0]*K+i)];
        }

        for ( int i=0; i<4; i+=blockDim.z )
        {
            sm_Aj[i+4][lane] = dev_A[I_MAP(blockIdx.z, blockIdx.x*slice+t*32+lane, index[1]*K+i)];
        }

        // do GEMM
        do_wmma_shared_tile<36,36>(sm_Ai, sm_Aj, 32, 0, 0, &frag_As);
    }

    // store data
    #define FRAG_MAP(y, x) ((y)*8+(x))
    if (lane < K*K) {
        int i = lane;
        int row = i/4;
        int col = i%4;
        dev_AiAi[O_MAP(blockIdx.z, blockIdx.y, blockIdx.x, row, col)] = frag_As.x[FRAG_MAP(col, row)];
    } else {
        int i = lane - 16;
        int row = i/4;
        int col = i%4;
        dev_AiAj[O_MAP(blockIdx.z, blockIdx.y, blockIdx.x, row, col)] = frag_As.x[FRAG_MAP(col, row+4)];
    }

    if (lane < K*K) {
        int i = lane - 16;
        int row = i/4;
        int col = i%4;
        dev_AjAj[O_MAP(blockIdx.z, blockIdx.y, blockIdx.x, row, col)] = frag_As.x[FRAG_MAP(col+4, row+4)];
    }
    

    #undef K
    #undef I_MAP
    #undef O_MAP
    #undef FRAG_MAP
}

// <<< (sliceNum, p, batch), (2, K/8, 32) >>>
template<int K>
__global__ void updateBlockColumn2_tc(double *dev_A, double *dev_V, double *dev_jointG, int *dev_pairsOfEVD, int p, int q, int height, int width, int slice)
{
    #define A_MAP(batch, y, x) ((batch)*height*width + (y) + (x)*height)
    #define V_MAP(batch, y, x) ((batch)*width*width + (y) + (x)*width)
    #define G_MAP(batch, pair, offset) ((batch)*4*K*K*p + (pair)*4*K*K + offset)

    using namespace nvcuda;

    // padding to aliviate conflict
    __shared__ double sm_A[32][K*2+4];
    __shared__ double sm_V[32][K*2+4];
    __shared__ double sm_G[K*2][K*2+4];

    __shared__ unsigned index[2];

    const int iter = slice / 32;
    const int lane = threadIdx.x;
    const int wx = threadIdx.y;
    const int wy = threadIdx.z;
    const int wx_size = blockDim.y;
    const int wy_size = blockDim.z;

    const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    const int num_warp = wx_size * wy_size;
    const int num_thd = num_warp << 5;

    if ( tid == 0 )
    {
        index[0] = dev_pairsOfEVD[2 * (blockIdx.z * p + blockIdx.y)];
        index[1] = dev_pairsOfEVD[2 * (blockIdx.z * p + blockIdx.y) + 1];
        // if ( blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 0) {
        //     printf("grid dim [%d, %d, %d] iters = %d\n", gridDim.z, gridDim.y, gridDim.x, iter);
        // } 
        // if (A_MAP(blockIdx.z, row_base + lane, col) == 32)
        //     printf("(%d, %d, %d): %d(i=%d) writing to A[32]\n", blockIdx.z, blockIdx.y, blockIdx.x, tid, i);

    }
    // __syncthreads();

    for ( int i = tid; i < 4*K*K; i += num_thd )
    {
        const int row = i/(K*2);
        const int col = i%(K*2);
        sm_G[row][col] = dev_jointG[G_MAP(blockIdx.z, blockIdx.y, i)]; //suoyi shuo suan affine map genben bushi ren gai gan de shi
    }
    // __syncthreads();
    // if ( blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 0 && tid == 0 )
    // {
    //     for (int i=0; i<2*K; i++){

    //         for (int j=0; j< 2*K; j++)
    //             printf("%.2f ", sm_G[i][j]);
    //         printf("\n");
    //     }
    //     printf("\n\n");

    //     for (int i=0; i<2*K; i++){

    //         for (int j=0; j< 2*K; j++)
    //             printf("%.2f ", dev_jointG[G_MAP(0,0,j*2*K+i)]);
    //         printf("\n");
    //     }
    // }
    __syncthreads();

    if (blockIdx.x * slice < width) { // involving V update
        for (int t = 0; t < iter; t++)
        {
            wmma::fragment<wmma::accumulator, 8, 8, 4, double> frag_As[4];
            wmma::fragment<wmma::accumulator, 8, 8, 4, double> frag_Vs[4];

            for (int i=0; i<4; ++i) {
                wmma::fill_fragment(frag_As[i], 0);
                wmma::fill_fragment(frag_Vs[i], 0);
            }

            const int row_base = blockIdx.x*slice + t*32;
            // load A tile
            for ( int i = wx; i < K; i += wx_size )
            {
                int col = index[wy] * K + i;
                sm_A[lane][i+wy*K] = dev_A[A_MAP(blockIdx.z, row_base + lane, col)];    
            }
            

            // load V tile conditionally
            for ( int i = wx; i < K; i += wx_size )
            {
                int col = index[wy] * K + i;
                sm_V[lane][i+wy*K] = (row_base + lane < width) ? 
                                dev_V[V_MAP(blockIdx.z, row_base + lane, col)]
                                : 0;
            }
            // __syncthreads();
            // if ( blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 0 && tid == 0 )
            // {
            //     printf("\n--------iter %d--------\n", t);
            //     for (int i=0; i<32; i++){

            //         for (int j=0; j< 2*K; j++)
            //             printf("%.2f ", sm_A[i][j]);
            //         printf("\n");
            //     }
            //     printf("\n\n");

            //     for (int i=0; i<32; i++){

            //         for (int j=0; j< 2*K; j++)
            //             printf("%.2f ", dev_A[A_MAP(0, row_base+i, j)]);
            //         printf("\n");
            //     }
            // }
            __syncthreads();

            do_wmma16x16_shared_tile<K*2+4, K*2+4>(sm_A, sm_G, 2*K, wy, wx, &frag_As[0]);
            do_wmma16x16_shared_tile<K*2+4, K*2+4>(sm_V, sm_G, 2*K, wy, wx, &frag_Vs[0]);
            
            __syncthreads();

            wmma::store_matrix_sync(&sm_A[16*wy  ][16*wx  ], frag_As[0], 2*K+4, wmma::mem_row_major);
            wmma::store_matrix_sync(&sm_A[16*wy  ][16*wx+8], frag_As[1], 2*K+4, wmma::mem_row_major);
            wmma::store_matrix_sync(&sm_A[16*wy+8][16*wx  ], frag_As[2], 2*K+4, wmma::mem_row_major);
            wmma::store_matrix_sync(&sm_A[16*wy+8][16*wx+8], frag_As[3], 2*K+4, wmma::mem_row_major);

            wmma::store_matrix_sync(&sm_V[16*wy  ][16*wx  ], frag_Vs[0], 2*K+4, wmma::mem_row_major);
            wmma::store_matrix_sync(&sm_V[16*wy  ][16*wx+8], frag_Vs[1], 2*K+4, wmma::mem_row_major);
            wmma::store_matrix_sync(&sm_V[16*wy+8][16*wx  ], frag_Vs[2], 2*K+4, wmma::mem_row_major);
            wmma::store_matrix_sync(&sm_V[16*wy+8][16*wx+8], frag_Vs[3], 2*K+4, wmma::mem_row_major);
            // __syncthreads();
            // if ( blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 0 && tid == 0 )
            // {
            //     printf("iter %d: frag_As[0,0] = %f\n", t, frag_As[0].x[0]);
            // }
            __syncthreads();
            
            // store A tile
            for ( int i = wx; i < K; i += wx_size )
            {
                int col = index[wy] * K + i;

                dev_A[A_MAP(blockIdx.z, row_base + lane, col)] = sm_A[lane][i+wy*K];
            }

            // store V tile conditionally
            for ( int i = wx; i < K; i += wx_size )
            {
                int col = index[wy] * K + i;
                if (row_base + lane < width) 
                    dev_V[V_MAP(blockIdx.z, row_base + lane, col)] = sm_V[lane][i+wy*K];
            }

            __syncthreads();
        }

    } else { // no V update
        for (int t = 0; t < iter; t++)
        {
            wmma::fragment<wmma::accumulator, 8, 8, 4, double> frag_As[4];
            for (int i=0; i<4; ++i) {
                wmma::fill_fragment(frag_As[i], 0);
            }

            const int row_base = blockIdx.x*slice + t*32;
            // load A tile
            for ( int i = wx; i < K; i += wx_size )
            {
                int col = index[wy] * K + i;

                sm_A[lane][i+wy*K] = dev_A[A_MAP(blockIdx.z, row_base + lane, col)];
            }
            __syncthreads();

            do_wmma16x16_shared_tile<K*2+4, K*2+4>(sm_A, sm_G, 2*K, wy, wx, &frag_As[0]);

            wmma::store_matrix_sync(&sm_A[16*wy  ][16*wx  ], frag_As[0], 2*K+4, wmma::mem_row_major);
            wmma::store_matrix_sync(&sm_A[16*wy  ][16*wx+8], frag_As[1], 2*K+4, wmma::mem_row_major);
            wmma::store_matrix_sync(&sm_A[16*wy+8][16*wx  ], frag_As[2], 2*K+4, wmma::mem_row_major);
            wmma::store_matrix_sync(&sm_A[16*wy+8][16*wx+8], frag_As[3], 2*K+4, wmma::mem_row_major);
            __syncthreads();

            // store A tile
            for ( int i = wx; i < K; i += wx_size )
            {
                int col = index[wy] * K + i;
                dev_A[A_MAP(blockIdx.z, row_base + lane, col)] = sm_A[lane][i+wy*K];
            }
            __syncthreads();
        }
    }
    #undef A_MAP
    #undef V_MAP
    #undef G_MAP
}

// <<< (sliceNum, p, batch), (2, K/8, 32) >>>
template<>
__global__ void updateBlockColumn2_tc<4>(double *dev_A, double *dev_V, double *dev_jointG, int *dev_pairsOfEVD, int p, int q, int height, int width, int slice)
{
    #define K 4
    #define A_MAP(batch, y, x) ((batch)*height*width + (y) + (x)*height)
    #define V_MAP(batch, y, x) ((batch)*width*width + (y) + (x)*width)
    #define G_MAP(batch, pair, offset) ((batch)*4*K*K*p + (pair)*4*K*K + offset)

    using namespace nvcuda;

    // padding to aliviate conflict
    __shared__ double sm_A[32][K*2+4];
    __shared__ double sm_V[32][K*2+4];
    __shared__ double sm_G[K*2][K*2+4];

    __shared__ unsigned index[2];

    const int iter = slice / 32;
    const int lane = threadIdx.x;
    const int wx = threadIdx.y;
    const int wy = threadIdx.z;
    const int wx_size = blockDim.y;
    const int wy_size = blockDim.z;

    const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    const int num_warp = wx_size * wy_size;
    const int num_thd = num_warp << 5;

    if ( tid == 0 )
    {
        index[0] = dev_pairsOfEVD[2 * (blockIdx.z * p + blockIdx.y)];
        index[1] = dev_pairsOfEVD[2 * (blockIdx.z * p + blockIdx.y) + 1];
        // if ( blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 0) {
        //     printf("grid dim [%d, %d, %d] iters = %d\n", gridDim.z, gridDim.y, gridDim.x, iter);
        // } 
        // if (A_MAP(blockIdx.z, row_base + lane, col) == 32)
        //     printf("(%d, %d, %d): %d(i=%d) writing to A[32]\n", blockIdx.z, blockIdx.y, blockIdx.x, tid, i);

    }
    // __syncthreads();

    for ( int i = tid; i < 4*K*K; i += num_thd )
    {
        const int row = i/(K*2);
        const int col = i%(K*2);
        sm_G[row][col] = dev_jointG[G_MAP(blockIdx.z, blockIdx.y, i)]; //suoyi shuo suan affine map genben bushi ren gai gan de shi
    }
    // __syncthreads();
    // if ( blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 0 && tid == 0 )
    // {
    //     for (int i=0; i<2*K; i++){

    //         for (int j=0; j< 2*K; j++)
    //             printf("%.2f ", sm_G[i][j]);
    //         printf("\n");
    //     }
    //     printf("\n\n");

    //     for (int i=0; i<2*K; i++){

    //         for (int j=0; j< 2*K; j++)
    //             printf("%.2f ", dev_jointG[G_MAP(0,0,j*2*K+i)]);
    //         printf("\n");
    //     }
    // }
    __syncthreads();

    if (blockIdx.x * slice < width) { // involving V update
        for (int t = 0; t < iter; t++)
        {
            wmma::fragment<wmma::accumulator, 8, 8, 4, double> frag_As[2];
            wmma::fragment<wmma::accumulator, 8, 8, 4, double> frag_Vs[2];

            for (int i=0; i<2; ++i) {
                wmma::fill_fragment(frag_As[i], 0);
                wmma::fill_fragment(frag_Vs[i], 0);
            }

            const int row_base = blockIdx.x*slice + t*32;
            // load A tile
            for ( int i = wx; i < K; i += wx_size )
            {
                int col = index[wy] * K + i;
                sm_A[lane][i+wy*K] = dev_A[A_MAP(blockIdx.z, row_base + lane, col)];    
            }
            

            // load V tile conditionally
            for ( int i = wx; i < K; i += wx_size )
            {
                int col = index[wy] * K + i;
                sm_V[lane][i+wy*K] = (row_base + lane < width) ? 
                                dev_V[V_MAP(blockIdx.z, row_base + lane, col)]
                                : 0;
            }
            // __syncthreads();
            // if ( blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 0 && tid == 0 )
            // {
            //     printf("\n--------iter %d--------\n", t);
            //     for (int i=0; i<32; i++){

            //         for (int j=0; j< 2*K; j++)
            //             printf("%.2f ", sm_A[i][j]);
            //         printf("\n");
            //     }
            //     printf("\n\n");

            //     for (int i=0; i<32; i++){

            //         for (int j=0; j< 2*K; j++)
            //             printf("%.2f ", dev_A[A_MAP(0, row_base+i, j)]);
            //         printf("\n");
            //     }
            // }
            __syncthreads();

            do_wmma16x8_shared_tile<K*2+4, K*2+4>(sm_A, sm_G, 2*K, wy, wx, &frag_As[0]);
            do_wmma16x8_shared_tile<K*2+4, K*2+4>(sm_V, sm_G, 2*K, wy, wx, &frag_Vs[0]);
            
            __syncthreads();

            wmma::store_matrix_sync(&sm_A[16*wy  ][16*wx  ], frag_As[0], 2*K+4, wmma::mem_row_major);
            wmma::store_matrix_sync(&sm_A[16*wy+8][16*wx  ], frag_As[1], 2*K+4, wmma::mem_row_major);

            wmma::store_matrix_sync(&sm_V[16*wy  ][16*wx  ], frag_Vs[0], 2*K+4, wmma::mem_row_major);
            wmma::store_matrix_sync(&sm_V[16*wy+8][16*wx  ], frag_Vs[1], 2*K+4, wmma::mem_row_major);
            // __syncthreads();
            // if ( blockIdx.z == 0 && blockIdx.y == 0 && blockIdx.x == 0 && tid == 0 )
            // {
            //     printf("iter %d: frag_As[0,0] = %f\n", t, frag_As[0].x[0]);
            // }
            __syncthreads();
            
            // store A tile
            for ( int i = wx; i < K; i += wx_size )
            {
                int col = index[wy] * K + i;

                dev_A[A_MAP(blockIdx.z, row_base + lane, col)] = sm_A[lane][i+wy*K];
            }

            // store V tile conditionally
            for ( int i = wx; i < K; i += wx_size )
            {
                int col = index[wy] * K + i;
                if (row_base + lane < width) 
                    dev_V[V_MAP(blockIdx.z, row_base + lane, col)] = sm_V[lane][i+wy*K];
            }

            __syncthreads();
        }

    } else { // no V update
        for (int t = 0; t < iter; t++)
        {
            wmma::fragment<wmma::accumulator, 8, 8, 4, double> frag_As[2];
            for (int i=0; i<4; ++i) {
                wmma::fill_fragment(frag_As[i], 0);
            }

            const int row_base = blockIdx.x*slice + t*32;
            // load A tile
            for ( int i = wx; i < K; i += wx_size )
            {
                int col = index[wy] * K + i;

                sm_A[lane][i+wy*K] = dev_A[A_MAP(blockIdx.z, row_base + lane, col)];
            }
            __syncthreads();

            do_wmma16x8_shared_tile<K*2+4, K*2+4>(sm_A, sm_G, 2*K, wy, wx, &frag_As[0]);

            wmma::store_matrix_sync(&sm_A[16*wy  ][16*wx  ], frag_As[0], 2*K+4, wmma::mem_row_major);
            wmma::store_matrix_sync(&sm_A[16*wy+8][16*wx  ], frag_As[1], 2*K+4, wmma::mem_row_major);
            __syncthreads();

            // store A tile
            for ( int i = wx; i < K; i += wx_size )
            {
                int col = index[wy] * K + i;
                dev_A[A_MAP(blockIdx.z, row_base + lane, col)] = sm_A[lane][i+wy*K];
            }
            __syncthreads();
        }
    }

}
