#include <cuda.h>
#include <mma.h>

namespace nvcuda {

using namespace wmma;
template <int LDA, int LDB>
__device__ __forceinline__ 
void do_wmma_shared_tile(double sm_A[][LDA], double sm_B[][LDB], const int tile_k, const int wy, const int wx,
                         fragment<accumulator, 8, 8, 4, double>* accum_frag)
{   
    fragment<matrix_a, 8, 8, 4, double, row_major> frag_a;
    fragment<matrix_b, 8, 8, 4, double, col_major> frag_b;

    for ( int k = 0; k < tile_k; k += 4 )
    {
        load_matrix_sync(frag_a, &sm_A[8*wy][k], LDA);
        load_matrix_sync(frag_b, &sm_B[8*wx][k], LDB);
        
        mma_sync(accum_frag[0], frag_a, frag_b, accum_frag[0]);
    }
}

template <int LDA, int LDB>
__device__ __forceinline__ 
void do_wmma16x16_shared_tile(double sm_A[][LDA], double sm_B[][LDB], const int tile_k, const int wy, const int wx,
                              fragment<accumulator, 8, 8, 4, double>* accum_frag)
{   
    fragment<matrix_a, 8, 8, 4, double, row_major> frag_a[2];
    fragment<matrix_b, 8, 8, 4, double, col_major> frag_b[2];

    for ( int k = 0; k < tile_k; k += 4 )
    {
        load_matrix_sync(frag_a[0], &sm_A[16*wy][k], LDA);
        load_matrix_sync(frag_a[1], &sm_A[16*wy+8][k], LDA);
        load_matrix_sync(frag_b[0], &sm_B[16*wx][k], LDB);
        load_matrix_sync(frag_b[1], &sm_B[16*wx+8][k], LDB);
        
        mma_sync(accum_frag[0], frag_a[0], frag_b[0], accum_frag[0]);
        mma_sync(accum_frag[1], frag_a[0], frag_b[1], accum_frag[1]);
        mma_sync(accum_frag[2], frag_a[1], frag_b[0], accum_frag[2]);
        mma_sync(accum_frag[3], frag_a[1], frag_b[1], accum_frag[3]);
    }
}

template <int LDA, int LDB>
__device__ __forceinline__ 
void do_wmma16x8_shared_tile(double sm_A[][LDA], double sm_B[][LDB], const int tile_k, const int wy, const int wx,
                              fragment<accumulator, 8, 8, 4, double>* accum_frag)
{   
    fragment<matrix_a, 8, 8, 4, double, row_major> frag_a[2];
    fragment<matrix_b, 8, 8, 4, double, col_major> frag_b;

    for ( int k = 0; k < tile_k; k += 4 )
    {
        load_matrix_sync(frag_a[0], &sm_A[16*wy][k], LDA);
        load_matrix_sync(frag_a[1], &sm_A[16*wy+8][k], LDA);
        load_matrix_sync(frag_b, &sm_B[8*wx][k], LDB);
        
        mma_sync(accum_frag[0], frag_a[0], frag_b, accum_frag[0]);
        mma_sync(accum_frag[1], frag_a[1], frag_b, accum_frag[1]);
    }
}


template <int LDA, int LDB>
__device__ __forceinline__ 
void do_wmma3_shared_tile(double sm_A[][LDA], double sm_B[][LDB], const int tile_k, const int wy, const int wx,
                          fragment<accumulator, 8, 8, 4, double> *accum_frag)
{   
    fragment<matrix_a, 8, 8, 4, double, row_major> frag_al, frag_bl;
    fragment<matrix_b, 8, 8, 4, double, col_major> frag_br, frag_ar;

    for ( int k = 0; k < tile_k; k += 4 )
    {
        load_matrix_sync(frag_al, &sm_A[8*wy][k], LDA);
        load_matrix_sync(frag_bl, &sm_B[8*wy][k], LDB);
        load_matrix_sync(frag_ar, &sm_A[8*wx][k], LDA);
        load_matrix_sync(frag_br, &sm_B[8*wx][k], LDB);
        
        mma_sync(accum_frag[0], frag_al, frag_ar, accum_frag[0]);
        mma_sync(accum_frag[1], frag_bl, frag_br, accum_frag[1]);
        mma_sync(accum_frag[2], frag_al, frag_br, accum_frag[2]);
    }

}

};