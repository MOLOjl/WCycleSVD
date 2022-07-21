#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <limits>
#include <string.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "magma_svd.hpp"

using namespace std;
using namespace std::chrono;

#define PRECISION 1e-7
#define k 16
#define slice 32

void print_matrix(double* m, int h, int w, string path){
    FILE* fp;
    fp = fopen(path.c_str(), "w");
    
	// fp = fopen("./result.txt", "a+");
    // 行优先
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            fprintf(fp, "%lf ", m[i * w + j]);
        }
        fprintf(fp, "\n");
    }
    
    // 列优先
    // for(int i=0; i<h; i++){
    //     for(int j=0; j<w; j++){
    //         fprintf(fp, "%lf ", m[j * h + i]);
    //     }
    //     fprintf(fp, "\n");
    // }
    
    fprintf(fp, "---------------------------\n");
    fclose(fp);
}

// hipcc svd.cpp -o svd -w
void generate_Matrix(int height, int width, FILE *fp)
{
	fp = fopen("input.txt", "w+");
	double temp;
	srand(time(NULL));
	for (int i = 0; i < height * width; i++)
	{
		temp = rand() % 100000 / (float)100000;
		fprintf(fp, "%lf ", temp);
	}
	fclose(fp);
	printf("A %dx%d matrix has been generated successfully in input.txt.\n", height, width);
}

__global__ void generate_roundRobin(int *dev_roundRobin, int n)
{ //  n = width / 2;
	__shared__ int firstLineOfRoundRobin[31];
	__shared__ int tempRoundRobin[31][31];
	if (threadIdx.y == 0)
	{
		if (threadIdx.x < n)
		{
			firstLineOfRoundRobin[threadIdx.x] = 2 * threadIdx.x;
			tempRoundRobin[0][threadIdx.x] = 2 * threadIdx.x;
		}
		else
		{
			firstLineOfRoundRobin[threadIdx.x] = 4 * n - 3 - 2 * threadIdx.x;
			tempRoundRobin[0][threadIdx.x] = 4 * n - 3 - 2 * threadIdx.x;
		}
	}
	__syncthreads();

	if (threadIdx.y != 0)
	{
		//dev_roundRobin[threadIdx.y * 2 * k + threadIdx.x] = firstLineOfRoundRobin[(threadIdx.x + 2 * k - 1 - threadIdx.y) % (2 * k - 1)];
		tempRoundRobin[threadIdx.y][threadIdx.x] = firstLineOfRoundRobin[(threadIdx.x + 2 * n - 1 - threadIdx.y) % (2 * n - 1)];
	}
	__syncthreads();
	if (threadIdx.x < n)
	{
		dev_roundRobin[threadIdx.y * 2 * n + threadIdx.x] = tempRoundRobin[threadIdx.y][threadIdx.x];
	}
	else
	{
		dev_roundRobin[threadIdx.y * 2 * n + threadIdx.x + 1] = tempRoundRobin[threadIdx.y][threadIdx.x];
	}
	if (threadIdx.x == 0)
	{
		dev_roundRobin[threadIdx.y * 2 * n + n] = 2 * n - 1;
	}
	__syncthreads();
}

//dim3 dimBlock0(width0 / 2, width0 / 2, 1);
__global__ void generate_roundRobin_64(int *dev_roundRobin, int n)
{ //  n = width / 2; 16-32
	__shared__ int firstLineOfRoundRobin[63];
	__shared__ int tempRoundRobin[63][63];
	if (threadIdx.y == 0)
	{
		firstLineOfRoundRobin[threadIdx.x] = 2 * threadIdx.x;
		tempRoundRobin[0][threadIdx.x] = 2 * threadIdx.x;
		if (threadIdx.x < (n - 1))
		{
			firstLineOfRoundRobin[threadIdx.x + n] = 2 * n - 3 - 2 * threadIdx.x;
			tempRoundRobin[0][threadIdx.x + n] = 2 * n - 3 - 2 * threadIdx.x;
		}
	}
	__syncthreads();

	if (threadIdx.y > 0)
	{
		//dev_roundRobin[threadIdx.y * 2 * k + threadIdx.x] = firstLineOfRoundRobin[(threadIdx.x + 2 * k - 1 - threadIdx.y) % (2 * k - 1)];
		tempRoundRobin[threadIdx.y][threadIdx.x] = firstLineOfRoundRobin[(threadIdx.x + 2 * n - 1 - threadIdx.y) % (2 * n - 1)];
		if (threadIdx.x < (n - 1))
		{
			tempRoundRobin[threadIdx.y][threadIdx.x + n] = firstLineOfRoundRobin[(threadIdx.x + n + 2 * n - 1 - threadIdx.y) % (2 * n - 1)];
		}
	}
	if (threadIdx.y < (n - 1))
	{
		tempRoundRobin[threadIdx.y + n][threadIdx.x] = firstLineOfRoundRobin[(threadIdx.x + 2 * n - 1 - threadIdx.y - n) % (2 * n - 1)];
		if (threadIdx.x < (n - 1))
		{
			tempRoundRobin[threadIdx.y + n][threadIdx.x + n] = firstLineOfRoundRobin[(threadIdx.x + n + 2 * n - 1 - threadIdx.y - n) % (2 * n - 1)];
		}
	}
	__syncthreads();

	dev_roundRobin[threadIdx.y * 2 * n + threadIdx.x] = tempRoundRobin[threadIdx.y][threadIdx.x];
	if (threadIdx.x < (n - 1))
	{
		dev_roundRobin[threadIdx.y * 2 * n + threadIdx.x + n + 1] = tempRoundRobin[threadIdx.y][threadIdx.x + n];
	}
	if (threadIdx.y < (n - 1))
	{
		dev_roundRobin[(threadIdx.y + n) * 2 * n + threadIdx.x] = tempRoundRobin[threadIdx.y + n][threadIdx.x];
		if (threadIdx.x < (n - 1))
		{
			dev_roundRobin[(threadIdx.y + n) * 2 * n + threadIdx.x + n + 1] = tempRoundRobin[threadIdx.y + n][threadIdx.x + n];
		}
	}
	if (threadIdx.x == 0)
	{
		dev_roundRobin[threadIdx.y * 2 * n + n] = 2 * n - 1;
		if (threadIdx.y < (n - 1))
		{
			dev_roundRobin[(threadIdx.y + n) * 2 * n + n] = 2 * n - 1;
		}
	}
	__syncthreads();
}

__device__ int p_a[2][2000];
__device__ int p1[2000];
__device__ int p_ab[2][2000][2000];
__device__ int pp[2000];
//dim3 dimGrid12(2 * p*q, batch, 1);
//generateDev_A<<<dimGrid12, 128>>> dev_A0->dev_A  32*4   k{4,8,16,24}
__global__ void generateDev_A(double *dev_A0, double *dev_A, double *dev_V, int height0, int width0, int height, int width, int p, int q)
{
	int bidxx = blockIdx.x % q;
	int bidxy = blockIdx.x / q;
	int bidy = blockIdx.y;
	int tidx = threadIdx.x % 32;
	int tidy = threadIdx.x / 32;
	int mynobs = width0;
	for (int i = 0; i < 4; i++)
	{
		if (((bidxy * k + 4 * i + tidy) < mynobs) && ((bidxx * 32 + tidx) < mynobs))
		{
			dev_A[bidy * height * width + ((bidxy * k + 4 * i + tidy) * height + bidxx * 32 + tidx)] = dev_A0[bidy * height0 * width0 + ((bidxy * k + 4 * i + tidy) * mynobs + bidxx * 32 + tidx)];
		}
		else
		{
			dev_A[bidy * height * width + ((bidxy * k + 4 * i + tidy) * height + bidxx * 32 + tidx)] = 0;
		}
		if (bidxx == 0 && tidx == 0 && (bidxy * k + 4 * i + tidy) < mynobs)
		{   
			dev_V[bidy * width * width+(bidxy * k + 4 * i + tidy) * width + bidxy * k + 4 * i + tidy] = 1;
		}
	}
}

//<< <2 * p*batch, 128 >> >
__global__ void computeFnorm1(double *dev_A, double *temp_Fnorm, int p, int q, int height, int width)
{
	double tmp = 0;
	int locx = threadIdx.x % 32;
	int locy = threadIdx.x / 32;
	int iter = k / 4;
	__shared__ double sm_Fnorm[4];
	for (int j = 0; j < q; j++)
	{
		for (int i = 0; i < iter; i++)
		{
			tmp += dev_A[(blockIdx.x * k + 4 * i + locy) * height + j * 32 + locx] * dev_A[(blockIdx.x * k + 4 * i + locy) * height + j * 32 + locx];
		}
	}

	for (int s = 16; s > 0; s /= 2)
	{
		tmp += __shfl_xor(tmp, s);
	}
	if (locx == 0)
		sm_Fnorm[locy] = tmp;
	__syncthreads();
	if (threadIdx.x == 0)
	{
		temp_Fnorm[blockIdx.x] = sm_Fnorm[0] + sm_Fnorm[1] + sm_Fnorm[2] + sm_Fnorm[3];
	}
}

__global__ void computeFnorm2(double *temp_Fnorm, double *Fnorm, int p)
{
	//extern __shared__ double fnorm[];
	double tmp = 0;
	int tid = threadIdx.x;
	for (int i = tid; i < 2 * p; i += 32)
	{
		tmp += temp_Fnorm[blockIdx.x * 2 * p + i];
	}
	for (int s = 16; s > 0; s /= 2)
	{
		tmp += __shfl_xor(tmp, s);
	}
	Fnorm[blockIdx.x] = sqrt(tmp);
}

__device__ void push_forward(int *p_a, int *p_b, int i, int length_b)
{
	volatile int temp;
	//__shared__ int p[2000];
	int tid;

	if (threadIdx.x == 0)
	{
		temp = p_a[length_b - i - 1];
		p_a[length_b - i - 1] = p_b[length_b - i - 1];
		p_b[length_b - i - 1] = temp;
		temp = p_b[0];
	}
	__syncthreads();

	tid = threadIdx.x;
	while (tid < length_b - 1)
	{
		pp[tid] = p_b[tid + 1];
		tid += blockDim.x;
	}
	__syncthreads();

	if (threadIdx.x == 0)
	{
		pp[length_b - 1] = temp;
	}
	__syncthreads();

	tid = threadIdx.x;
	while (tid < length_b)
	{
		p_b[tid] = pp[tid];
		tid += blockDim.x;
	}
	__syncthreads();
}

__global__ void getRankNewNew(int row)
{

	unsigned tid = threadIdx.x;
	unsigned iter = blockDim.x;

	for (unsigned i = tid; i < row; i += iter)
	{

		if (i % 2 == 0)
		{
			p_ab[0][i / 2][0] = i;
			p_a[0][i / 2] = i;
		}
		else
		{
			p_ab[1][i / 2][0] = i;
			p_a[1][i / 2] = i;
		}
	}
	__syncthreads();

	int count = 0;
	for (unsigned i = 1; i < row - (1 - row % 2); i++)
	{

		if (i - 1 == 2 * (count + 1))
		{
			count = count + 1;
		}
		__syncthreads();

		push_forward(p_a[0], p_a[1], count, row / 2);

		__syncthreads();

		for (unsigned j = tid; j < row / 2; j += iter)
		{
			p_ab[0][j][i] = p_a[0][j];
			p_ab[1][j][i] = p_a[1][j];
		}
		__syncthreads();
	}
}

//<< < 2 * p*batch, 128 >> >
__global__ void compute_norm(double *dev_A, double *dev_norm, unsigned int *dev_order, int height, int width, int p, int q)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int locx = tid % 32;
	int locy = tid / 32;
	int iter = k / 4;
	__shared__ double sm_norm[4];
	if (tid == 0)
	{
		dev_order[bid] = bid % (2 * p);
	}
	__syncthreads();
	double tmp = 0;
	for (int j = 0; j < q; j++)
	{
		for (int i = 0; i < iter; i++)
		{
			tmp += dev_A[(bid * k + 4 * i + locy) * height + j * 32 + locx] * dev_A[(bid * k + 4 * i + locy) * height + j * 32 + locx];
		}
	}
	for (int s = 16; s > 0; s /= 2)
	{
		tmp += __shfl_xor( tmp, s);
	}
	if (locx == 0)
		sm_norm[locy] = tmp;
	__syncthreads();
	if (tid == 0)
	{
		dev_norm[bid] = sm_norm[0] + sm_norm[1] + sm_norm[2] + sm_norm[3];
	}
}

//<< <batch, 1024>>>(dev_norm, dev_order, 2 * p, p);
__global__ void binoticSort_original(double *value, unsigned int *arr1, unsigned int len, int p)
{
	__shared__ double buf[1024];
	__shared__ unsigned int buf_index1[1024];
	int bid = blockIdx.x;
	buf[threadIdx.x] = (threadIdx.x < len ? value[bid * 2 * p + threadIdx.x] : 0xffffffffu * 1.0);
	buf_index1[threadIdx.x] = (threadIdx.x < len ? arr1[bid * 2 * p + threadIdx.x] : 0);
	__syncthreads();

	for (unsigned kk = 2; kk <= blockDim.x; kk *= 2)
	{ //buid k elements ascend or descend
		for (unsigned jj = kk >> 1; jj > 0; jj >>= 1)
		{ //merge longer binotic into shorter binotic
			unsigned swapIdx = threadIdx.x ^ jj;
			double myelem = buf[threadIdx.x];
			double other = buf[swapIdx];
			unsigned int myindex1 = buf_index1[threadIdx.x];

			__syncthreads();

			unsigned ascend = kk * (swapIdx < threadIdx.x);
			unsigned descend = kk * (swapIdx > threadIdx.x);
			//if swapIdx > threadIdx.x, then ascend = 0, descend = k;
			//if swapIdx < threadIdx.x, then ascend = k, descend = 0;

			bool swap = false;
			// threadIdx.x & k == 0 or k;
			if ((threadIdx.x & kk) == ascend)
			{
				if (myelem > other)
					swap = true;
			}
			if ((threadIdx.x & kk) == descend)
			{
				if (myelem < other)
					swap = true;
			}

			if (swap)
			{
				buf[swapIdx] = myelem;
				buf_index1[swapIdx] = myindex1;
			}
			__syncthreads();
		}
	}

	if (threadIdx.x < len)
	{
		value[bid * 2 * p + threadIdx.x] = buf[threadIdx.x];
		arr1[bid * 2 * p + threadIdx.x] = buf_index1[threadIdx.x];
	}
	__syncthreads();
}

void parallel_ordering_choice(int p, int q, unsigned int *dev_order, double *dev_A, int height, int width, double *dev_norm, int batch)
{

	//compute_norm<<<2 * p * batch, 128>>>(dev_A, dev_norm, dev_order, height, width, p, q);
	hipLaunchKernelGGL(compute_norm, dim3(2 * p*batch), dim3(128), 0, 0, dev_A, dev_norm, dev_order, height, width, p, q);
	//binoticSort_original<<<batch, 1024>>>(dev_norm, dev_order, 2 * p, p);
	hipLaunchKernelGGL(binoticSort_original, dim3(batch), dim3(1024), 0, 0, dev_norm, dev_order, 2 * p, p);
	hipDeviceSynchronize();
}

bool ifallpass(unsigned *host_allpass, int batch, int p)
{
	int count = 0;
	for (int i = 0; i < batch; i++)
	{
		if (host_allpass[i] == p)
			count++;
	}
	if (count == batch)
		return true;
	else
		return false;
}

//dim3 dimGrid77(sliceNum, p, batch);
//<< < dimGrid77, 256 >> >
__global__ void generate_jointG00(double *dev_A, int height, int width, unsigned int *dev_order, unsigned int *dev_pass, int p, int q, int *dev_pairsOfEVD,
								  double *dev_AiAi, double *dev_AiAj, double *dev_AjAj, int iterNum, int sliceNum)
{
	__shared__ int index[2]; //index[1]>index[0]
	__shared__ double sm_Ai[32][24];
	__shared__ double sm_Aj[32][24];
	int tid = threadIdx.x;
	int tid1 = threadIdx.x + 256;
	int tid2 = threadIdx.x + 512;
	int iter = slice / 32;

	if (tid == 0)
	{
		index[0] = dev_order[(2 * p - 1) - p_ab[0][blockIdx.y][iterNum] + blockIdx.z * 2 * p];
		index[1] = dev_order[(2 * p - 1) - p_ab[1][blockIdx.y][iterNum] + blockIdx.z * 2 * p];
		dev_pairsOfEVD[2 * (blockIdx.z * p + blockIdx.y)] = index[0];
		dev_pairsOfEVD[2 * (blockIdx.z * p + blockIdx.y) + 1] = index[1];
	}

	__syncthreads();

	double Cvalueii = 0.0;
	double Cvalueij = 0.0;
	double Cvaluejj = 0.0;
	double Cvalueii1 = 0.0;
	double Cvalueij1 = 0.0;
	double Cvaluejj1 = 0.0;
	double Cvalueii2 = 0.0;
	double Cvalueij2 = 0.0;
	double Cvaluejj2 = 0.0;
	int locx = tid % 32; //32*8=256
	int locy = tid / 32;

	for (int t = 0; t < iter; t++)
	{
		if ((blockIdx.x * iter + t) < q)
		{
			if (tid < 128)
			{
				for (int i = 0; i < k / 4; i++)
				{
					sm_Ai[locx][locy + i * 4] = dev_A[blockIdx.z * height * width + (index[0] * k + locy + i * 4) * height + blockIdx.x * slice + t * 32 + locx];
				}
			}
			else
			{
				for (int i = 0; i < k / 4; i++)
				{
					sm_Aj[locx][locy - 4 + i * 4] = dev_A[blockIdx.z * height * width + (index[1] * k + locy - 4 + i * 4) * height + blockIdx.x * slice + t * 32 + locx];
				}
			}

			__syncthreads();
			////////////////////////////////// 4*4 24*24
			int locxx, locyy;
			if (tid < k * k)
			{
				locxx = tid % k;
				locyy = tid / k;
				for (unsigned i = 0; i < 32; i++)
				{
					Cvalueii += sm_Ai[i][locxx] * sm_Ai[i][locyy];
					Cvalueij += sm_Ai[i][locxx] * sm_Aj[i][locyy];
					Cvaluejj += sm_Aj[i][locxx] * sm_Aj[i][locyy];
				}
			}
			__syncthreads();
			if (tid1 < k * k)
			{
				locxx = tid1 % k;
				locyy = tid1 / k;
				for (unsigned i = 0; i < 32; i++)
				{
					Cvalueii1 += sm_Ai[i][locxx] * sm_Ai[i][locyy];
					Cvalueij1 += sm_Ai[i][locxx] * sm_Aj[i][locyy];
					Cvaluejj1 += sm_Aj[i][locxx] * sm_Aj[i][locyy];
				}
			}
			__syncthreads();
			if (tid2 < k * k)
			{
				locxx = tid2 % k;
				locyy = tid2 / k;
				for (unsigned i = 0; i < 32; i++)
				{
					Cvalueii2 += sm_Ai[i][locxx] * sm_Ai[i][locyy];
					Cvalueij2 += sm_Ai[i][locxx] * sm_Aj[i][locyy];
					Cvaluejj2 += sm_Aj[i][locxx] * sm_Aj[i][locyy];
				}
			}
			__syncthreads();
		}
		__syncthreads();
	}
	__syncthreads();
	if (tid < k * k)
	{
		dev_AiAi[blockIdx.z * k * k * sliceNum * p + blockIdx.y * k * k * sliceNum + blockIdx.x * k * k + tid] = Cvalueii;
		dev_AiAj[blockIdx.z * k * k * sliceNum * p + blockIdx.y * k * k * sliceNum + blockIdx.x * k * k + tid] = Cvalueij;
		dev_AjAj[blockIdx.z * k * k * sliceNum * p + blockIdx.y * k * k * sliceNum + blockIdx.x * k * k + tid] = Cvaluejj;
	}
	if (tid1 < k * k)
	{
		dev_AiAi[blockIdx.z * k * k * sliceNum * p + blockIdx.y * k * k * sliceNum + blockIdx.x * k * k + tid1] = Cvalueii1;
		dev_AiAj[blockIdx.z * k * k * sliceNum * p + blockIdx.y * k * k * sliceNum + blockIdx.x * k * k + tid1] = Cvalueij1;
		dev_AjAj[blockIdx.z * k * k * sliceNum * p + blockIdx.y * k * k * sliceNum + blockIdx.x * k * k + tid1] = Cvaluejj1;
	}
	if (tid2 < k * k)
	{
		dev_AiAi[blockIdx.z * k * k * sliceNum * p + blockIdx.y * k * k * sliceNum + blockIdx.x * k * k + tid2] = Cvalueii2;
		dev_AiAj[blockIdx.z * k * k * sliceNum * p + blockIdx.y * k * k * sliceNum + blockIdx.x * k * k + tid2] = Cvalueij2;
		dev_AjAj[blockIdx.z * k * k * sliceNum * p + blockIdx.y * k * k * sliceNum + blockIdx.x * k * k + tid2] = Cvaluejj2;
	}
}

//dim3 dimGrid7(p, batch, 1);
//<< < dimGrid7, 256 >> >
__global__ void generate_jointG21(double *dev_jointG, double *dev_AiAi, double *dev_AiAj, double *dev_AjAj, double *dev_Fnorm, unsigned int *dev_pass, int p, int sliceNum)
{
	int tid = threadIdx.x;
	int locx;
	int locy;
	__shared__ double sum[8];
	double d = 0;
	double value1 = 0;
	double value2 = 0;
	double value3 = 0;
	double tmp = 0;

	if (k <= 16)
	{
		locx = tid % k;
		locy = tid / k;
		if (tid < k * k)
		{
			for (int i = 0; i < sliceNum; i++)
			{
				tmp = dev_AiAj[blockIdx.y * k * k * sliceNum * p + blockIdx.x * k * k * sliceNum + i * k * k + tid];
				value1 += tmp;
			}
			d = value1 * value1;
		}
		for (int s = 16; s > 0; s /= 2)
		{
			d += __shfl_xor( d, s);
		}
		if (tid < k * k)
		{
			if (tid % 32 == 0)
				sum[tid / 32] = d;
		}
		__syncthreads();

		if (tid == 0)
		{
			for (int i = 1; i < k * k / 32; i++)
			{
				sum[0] += sum[i];
			}
			//printf(" mat %d:%.15lf \n", blockIdx.y , sqrt(sum[0]));
			if (sqrt(sum[0]) > PRECISION * dev_Fnorm[blockIdx.y] * dev_Fnorm[blockIdx.y])
			{
				dev_pass[p * blockIdx.y + blockIdx.x] = 0;
			}
		}
		__syncthreads();
		if (tid < k * k)
		{
			for (int i = 0; i < sliceNum; i++)
			{
				value2 += dev_AiAi[blockIdx.y * k * k * sliceNum * p + blockIdx.x * k * k * sliceNum + i * k * k + tid];
				value3 += dev_AjAj[blockIdx.y * k * k * sliceNum * p + blockIdx.x * k * k * sliceNum + i * k * k + tid];
			}
			dev_jointG[blockIdx.y * 2 * k * 2 * k * p + (blockIdx.x * 2 * k + locy) * 2 * k + locx] = value2;
			dev_jointG[blockIdx.y * 2 * k * 2 * k * p + (blockIdx.x * 2 * k + k + locy) * 2 * k + locx] = value1;
			dev_jointG[blockIdx.y * 2 * k * 2 * k * p + (blockIdx.x * 2 * k + locx) * 2 * k + k + locy] = value1;
			dev_jointG[blockIdx.y * 2 * k * 2 * k * p + (blockIdx.x * 2 * k + k + locy) * 2 * k + locx + k] = value3;
		}
	}
	else
	{
		if (tid < 8)
			sum[tid] = 0;
		__syncthreads();
		for (int j = tid; j < k * k; j += 256)
		{
			locx = j % k;
			locy = j / k;
			value1 = 0;
			value2 = 0;
			value3 = 0;
			for (int i = 0; i < sliceNum; i++)
			{
				tmp = dev_AiAj[blockIdx.y * k * k * sliceNum * p + blockIdx.x * k * k * sliceNum + i * k * k + j];
				value1 += tmp;
			}
			d = value1 * value1;
			for (int s = 16; s > 0; s /= 2)
			{
				d += __shfl_xor(d, s);
			}
			if (tid % 32 == 0)
				sum[tid / 32] += d;
			for (int i = 0; i < sliceNum; i++)
			{
				value2 += dev_AiAi[blockIdx.y * k * k * sliceNum * p + blockIdx.x * k * k * sliceNum + i * k * k + j];
				value3 += dev_AjAj[blockIdx.y * k * k * sliceNum * p + blockIdx.x * k * k * sliceNum + i * k * k + j];
			}
			dev_jointG[blockIdx.y * 2 * k * 2 * k * p + (blockIdx.x * 2 * k + locy) * 2 * k + locx] = value2;
			dev_jointG[blockIdx.y * 2 * k * 2 * k * p + (blockIdx.x * 2 * k + k + locy) * 2 * k + locx] = value1;
			dev_jointG[blockIdx.y * 2 * k * 2 * k * p + (blockIdx.x * 2 * k + locx) * 2 * k + k + locy] = value1;
			dev_jointG[blockIdx.y * 2 * k * 2 * k * p + (blockIdx.x * 2 * k + k + locy) * 2 * k + locx + k] = value3;
			__syncthreads();
		}

		if (tid == 0)
		{
			for (int i = 1; i < 8; i++)
			{
				sum[0] += sum[i];
			}
			//printf(" mat %d:%.15lf \n", blockIdx.y, sqrt(sum[0]));
			if (sqrt(sum[0]) > 10.0 * PRECISION * dev_Fnorm[blockIdx.y])
			{
				dev_pass[p * blockIdx.y + blockIdx.x] = 0;
			}
		}
		__syncthreads();
	}
}

//int sliceNum = (height - 1) / slice + 1;
//dim3 dimGrid11(sliceNum, p, batch);
//updateBlockColumn2<<< dimGrid11, 256 >> >
__global__ void updateBlockColumn2_16(double *dev_A, double *dev_V, double *dev_jointG, int *dev_pairsOfEVD, int p, int q, int height, int width)
{
	__shared__ double sm_A[32 * 16 * 2];
	__shared__ double sm_V[32 * 16 * 2];
	__shared__ double sm_G[32][32];
	__shared__ unsigned index[2];
	int iter = slice / 32;
	int tid = threadIdx.x;
	int locx, locy;
	locx = tid % 32;
	locy = tid / 32;

	if (tid == 0)
	{
		index[0] = dev_pairsOfEVD[2 * (blockIdx.z * p + blockIdx.y)];
		index[1] = dev_pairsOfEVD[2 * (blockIdx.z * p + blockIdx.y) + 1];
	}
	for (int i = tid; i < 1024; i += 256)
	{
		sm_G[i / 32][i % 32] = dev_jointG[(blockIdx.z * p + blockIdx.y) * 1024 + i];
	}
	__syncthreads();

	double Avalue1 = 0.0;
	double Avalue11 = 0.0;
	double Avalue2 = 0.0;
	double Avalue22 = 0.0;
	double Ivalue1 = 0.0;
	double Ivalue11 = 0.0;
	double Ivalue2 = 0.0;
	double Ivalue22 = 0.0;

	for (int t = 0; t < iter; t++)
	{

		if ((blockIdx.x * iter + t) < q)
		{

			sm_A[tid] = dev_A[blockIdx.z * height * width + (index[0] * k + locy) * height + blockIdx.x * slice + t * 32 + locx];
			sm_A[tid + 256] = dev_A[blockIdx.z * height * width + (index[0] * k + locy + 8) * height + blockIdx.x * slice + t * 32 + locx];
			sm_A[tid + 512] = dev_A[blockIdx.z * height * width + (index[1] * k + locy) * height + blockIdx.x * slice + t * 32 + locx];
			sm_A[tid + 768] = dev_A[blockIdx.z * height * width + (index[1] * k + locy + 8) * height + blockIdx.x * slice + t * 32 + locx];
			if ((blockIdx.x * slice + t * 32 + locx) < width)
			{
				sm_V[tid] = dev_V[blockIdx.z * width * width + (index[0] * k + locy) * width + blockIdx.x * slice + t * 32 + locx];
				sm_V[tid + 256] = dev_V[blockIdx.z * width * width + (index[0] * k + locy + 8) * width + blockIdx.x * slice + t * 32 + locx];
				sm_V[tid + 512] = dev_V[blockIdx.z * width * width + (index[1] * k + locy) * width + blockIdx.x * slice + t * 32 + locx];
				sm_V[tid + 768] = dev_V[blockIdx.z * width * width + (index[1] * k + locy + 8) * width + blockIdx.x * slice + t * 32 + locx];
			}
			else
			{
				sm_V[tid] = 0;
				sm_V[tid + 256] = 0;
				sm_V[tid + 512] = 0;
				sm_V[tid + 768] = 0;
			}

			__syncthreads();
			Avalue1 = 0.0;
			Avalue11 = 0.0;
			Avalue2 = 0.0;
			Avalue22 = 0.0;
			Ivalue1 = 0.0;
			Ivalue11 = 0.0;
			Ivalue2 = 0.0;
			Ivalue22 = 0.0;
			for (unsigned j = 0; j < 2 * k; j++)
			{
				Avalue1 += sm_A[locx + 32 * j] * sm_G[locy][j];
				Avalue11 += sm_A[locx + 32 * j] * sm_G[locy + 8][j];
				Avalue2 += sm_A[locx + 32 * j] * sm_G[locy + 16][j];
				Avalue22 += sm_A[locx + 32 * j] * sm_G[locy + 24][j];
			}
			if ((blockIdx.x * slice + t * 32 + locx) < width)
			{
				for (unsigned j = 0; j < 2*k; j++)
				{
					Ivalue1 += sm_V[locx + 32 * j] * sm_G[locy][j];
					Ivalue11 += sm_V[locx + 32 * j] * sm_G[locy + 8][j];
					Ivalue2 += sm_V[locx + 32 * j] * sm_G[locy + 16][j];
					Ivalue22 += sm_V[locx + 32 * j] * sm_G[locy + 24][j];
				}
			}
			__syncthreads();

			dev_A[blockIdx.z * height * width + (index[0] * k + locy) * height + blockIdx.x * slice + t * 32 + locx] = Avalue1;
			dev_A[blockIdx.z * height * width + (index[0] * k + locy + 8) * height + blockIdx.x * slice + t * 32 + locx] = Avalue11;
			dev_A[blockIdx.z * height * width + (index[1] * k + locy) * height + blockIdx.x * slice + t * 32 + locx] = Avalue2;
			dev_A[blockIdx.z * height * width + (index[1] * k + locy + 8) * height + blockIdx.x * slice + t * 32 + locx] = Avalue22;
			if ((blockIdx.x * slice + t * 32 + locx) < width)
			{
				dev_V[blockIdx.z * width * width + (index[0] * k + locy) * width + blockIdx.x * slice + t * 32 + locx] = Ivalue1;
				dev_V[blockIdx.z * width * width + (index[0] * k + locy + 8) * width + blockIdx.x * slice + t * 32 + locx] = Ivalue11;
				dev_V[blockIdx.z * width * width + (index[1] * k + locy) * width + blockIdx.x * slice + t * 32 + locx] = Ivalue2;
				dev_V[blockIdx.z * width * width + (index[1] * k + locy + 8) * width + blockIdx.x * slice + t * 32 + locx] = Ivalue22;
			}
			__syncthreads();
		}
		__syncthreads();
	}
}

__global__ void myevd_batched(double *dev_jointG, int *dev_roundRobin, int p)
{
	__shared__ double shared_G[32][32];
	__shared__ int shared_roundRobin[31][32];
	__shared__ int step;
	__shared__ double shared_V[32][32];
	__shared__ double shared_operators[2][32];
	__shared__ int shared_pairs[2][32];
	shared_G[threadIdx.x][threadIdx.y] = dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + threadIdx.y * 2 * k + threadIdx.x];

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
						signTao = 0;
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

//void EVD_(double* dev_jointG, double* dev_A, double* dev_V, int* dev_pairsOfEVD, double* dev_S, int* dev_info, cusolverDnHandle_t cusolverH, cusolverEigMode_t jobz,
//	double *dev_work, int lwork, cublasFillMode_t uplo, syevjInfo_t syevj_params, int p, int q, int height, int width, int batch, int k) {
void EVD_(double *dev_jointG, double *dev_A, double *dev_V, int *dev_pairsOfEVD, int p, int q, int height, int width, int *dev_roundRobin, int batch, int sliceNum)
{
	//clock_t start_cpu2, stop_cpu2;

	dim3 dimGrid9(p, batch, 1);
	dim3 dimBlock9(2 * k, 2 * k, 1);
	//myevd_batched<<<dimGrid9, dimBlock9>>>(dev_jointG, dev_roundRobin, p);
	hipLaunchKernelGGL(myevd_batched, dimGrid9, dimBlock9, 0, 0, dev_jointG, dev_roundRobin, p);
	dim3 dimGrid11(sliceNum, p, batch);
	//updateBlockColumn2_16<<<dimGrid11, 256>>>(dev_A, dev_V, dev_jointG, dev_pairsOfEVD, p, q, height, width);
	hipLaunchKernelGGL(updateBlockColumn2_16, dimGrid11, dim3(256), 0, 0, dev_A, dev_V, dev_jointG, dev_pairsOfEVD, p, q, height, width);
}

__global__ void judgeFunc(unsigned *dev_allpass, unsigned *dev_pass, int length)
{

	__shared__ unsigned pass[1024];
	unsigned myid = threadIdx.x;

	pass[threadIdx.x] = (threadIdx.x < length ? dev_pass[blockIdx.x * length + threadIdx.x] : 0);
	__syncthreads();

	for (unsigned i = 2; i <= 1024; i *= 2)
	{

		if (myid < 1024 / i)
		{
			unsigned anotherOne = myid + 1024 / i;
			pass[myid] += pass[anotherOne];
		}
		__syncthreads();
	}

	if (myid < length)
	{
		dev_pass[blockIdx.x * length + myid] = 1;
	}

	if (myid == 0)
	{
		dev_allpass[blockIdx.x] = pass[0];
	}
}

//dim3 dimGrid10(2 * p, batch, 1);
//dim3 dimBlock10(32, k, 1);(dev_A, dev_U, dev_V, dev_V0, height, width, height0, width0, p, q, dev_diag, width0,k);
__global__ void getUDV(double *dev_A, double *dev_U, double *dev_I, double *dev_V, int height, int width, int height0, int width0, int p, int q, double *dev_diag)
{
	__shared__ double shared_A[32][16];
	__shared__ double sqrtSum[16];
	int mynobs = width0;
	double temp1 = 0;
	for (int j = 0; j < q; j++)
	{
		shared_A[threadIdx.x][threadIdx.y] = dev_A[blockIdx.y * height * width + (blockIdx.x * k + threadIdx.y) * height + j * 32 + threadIdx.x];
		__syncthreads();
		if (threadIdx.y == 0 && threadIdx.x < k)
		{
			for (int i = 0; i < 32; i++)
			{
				temp1 += shared_A[i][threadIdx.x] * shared_A[i][threadIdx.x];
			}
		}
		__syncthreads();
	}
	if (threadIdx.y == 0 && threadIdx.x < k)
	{
		sqrtSum[threadIdx.x] = sqrt(temp1);
		if ((blockIdx.x * k + threadIdx.x) < mynobs)
		{
			dev_diag[blockIdx.y * 1000 + blockIdx.x * k + threadIdx.x] = sqrtSum[threadIdx.x];
		}
	}
	__syncthreads();
	///get U
	double temp2;
	for (int j = 0; j < q; j++)
	{
		shared_A[threadIdx.x][threadIdx.y] = dev_A[blockIdx.y * height * width + (blockIdx.x * k + threadIdx.y) * height + j * 32 + threadIdx.x];
		__syncthreads();
		if (sqrtSum[threadIdx.y] != 0)
		{
			temp2 = shared_A[threadIdx.x][threadIdx.y] / sqrtSum[threadIdx.y];
		}
		else
		{
			temp2 = 0;
		}
		if ((j * 32 + threadIdx.x) < mynobs && (blockIdx.x * k + threadIdx.y) < mynobs)
		{
			dev_U[blockIdx.y * height0 * height0 + (blockIdx.x * k + threadIdx.y) * mynobs + j * 32 + threadIdx.x] = temp2;
		}
		__syncthreads();
	}
	__syncthreads();
	////get V
	for (int j = 0; j < q; j++)
	{
		if ((j * 32 + threadIdx.x) < mynobs && (blockIdx.x * k + threadIdx.y) < mynobs)
		{   //if(blockIdx.y==1) printf("%lf",dev_I[blockIdx.y * width * width + (blockIdx.x * k + threadIdx.y) * width + j * 32 + threadIdx.x]);
			dev_V[blockIdx.y * width0 * width0 + (blockIdx.x * k + threadIdx.y) * mynobs + j * 32 + threadIdx.x] = dev_I[blockIdx.y * width * width + (blockIdx.x * k + threadIdx.y) * width + j * 32 + threadIdx.x];
		}
	}
}

__global__ void trans(double *dev_A, int height, int width, int batch)
{
	int nCol = blockIdx.y * blockDim.y + threadIdx.y;
	int nRow = blockIdx.x * blockDim.x + threadIdx.x;
	double t;
	if (nRow < height && nCol < width)
	{
		t = dev_A[blockIdx.z * height * width + nCol * height + nRow];
	}
	__syncthreads();
	if (nRow < height && nCol < width)
	{
		dev_A[blockIdx.z * height * width + nRow * width + nCol] = t;
	}
}

double svd_large_matrix(double* host_A, int* shape){
    int height0 = shape[1];
	int width0 = shape[2];
	int batch = shape[0];

	// double *host_A;
	// host_A = (double *)malloc(sizeof(double) * height0 * width0);

	double *dev_A0, *dev_V, *dev_U, *dev_diag;
	int *dev_roundRobin;

	hipMalloc((void **)&dev_A0, sizeof(double) * height0 * width0 * batch);
	hipMalloc((void**)&dev_diag, sizeof(double)* height0 * batch);
	hipMalloc((void**)&dev_U, sizeof(double)* height0 * height0*batch);

	hipMemset(dev_U, 0, sizeof(double)* height0 * height0*batch);
	hipMemset(dev_V, 0, sizeof(double)* width0 * width0*batch);

	for (int i = 0;i < batch;i++) {
		hipMemcpy(dev_A0 + height0 * width0*i, host_A, sizeof(double)* height0 * width0, hipMemcpyHostToDevice);
	}
	
	int tag = 0;//0:height0>=width0; 1:height0<width0
	if (height0 < width0) {
		tag = 1;
		dim3 gird((height0 - 1) / 32 + 1, (width0 - 1) / 32 + 1, batch);
		dim3 block(32, 32);
		// trans<<<gird, block >>>(dev_A0, height0, width0, batch);
		hipLaunchKernelGGL(trans, gird, block, 0, 0, dev_A0, height0, width0, batch);
		hipDeviceSynchronize();
		int t = height0;
		height0 = width0;
		width0 = t;
	}
		
	int p = (width0 - 1) / (2 * k) + 1;
	int q = (height0 - 1) / slice + 1;

	int width = p * (2 * k);
	int height = q * slice;
	int sliceNum = q;

	double *dev_A, *dev_jointG, *dev_AiAi, *dev_AiAj, *dev_AjAj, *dev_V0;
	int  *dev_pairsOfEVD;
	
	hipMalloc((void**)&dev_A, sizeof(double)* height * width*batch);
	hipMalloc((void**)&dev_V, sizeof(double)* width * width*batch);
	hipMalloc((void**)&dev_U, sizeof(double)* height0 * height0*batch);
	hipMalloc((void**)&dev_V0, sizeof(double)* width0 * width0*batch);
	hipMemset(dev_V, 0, sizeof(double)* width * width*batch);
	hipMemset(dev_U, 0, sizeof(double)* height0 * height0*batch);

	// this way to test time doesn't work
	// clock_t start_cpu, stop_cpu;
	// start_cpu = clock();
	steady_clock::time_point t1 = steady_clock::now();

	hipMalloc((void**)&dev_roundRobin, sizeof(int) * (2 * k - 1) * 2 * k);
	hipMalloc((void**)&dev_pairsOfEVD, sizeof(int) * 2 * p*batch);
	hipMalloc((void**)&dev_jointG, sizeof(double) * 2 * k * 2 * k *p*batch);
	hipMalloc((void**)&dev_AiAi, sizeof(double)  * k *k*sliceNum*p*batch);
	hipMalloc((void**)&dev_AiAj, sizeof(double)  * k *k*sliceNum*p*batch);
	hipMalloc((void**)&dev_AjAj, sizeof(double)  * k *k*sliceNum*p*batch);
	hipMemset(dev_pairsOfEVD, 0, sizeof(int) * 2 * p*batch);

	unsigned *host_allpass = (unsigned*)malloc(sizeof(unsigned)*batch);
	unsigned *host_pass = (unsigned*)malloc(sizeof(unsigned)*p*batch);
	unsigned *dev_allpass;
	unsigned *dev_pass;
	hipMalloc((void**)&dev_allpass, sizeof(unsigned)*batch);
	hipMalloc((void**)&dev_pass, sizeof(unsigned)*p*batch);
	memset(host_pass, 0, sizeof(unsigned)*p*batch);
	hipMemset(dev_pass, 0, sizeof(unsigned)*p*batch);

	hipDeviceSynchronize();
	unsigned int len = 2 * p * (2 * p - 1) / 2;
	double *value = (double*)malloc(sizeof(double)*len);
	unsigned int *arr1 = (unsigned int*)malloc(sizeof(unsigned int)*len);
	unsigned int *arr2 = (unsigned int*)malloc(sizeof(unsigned int)*len);
	unsigned int *pairs = (unsigned int*)malloc(sizeof(unsigned int) * 2 * p);

	double *dev_norm;
	unsigned int *dev_order;
	hipMalloc((void**)&dev_norm, sizeof(double) * 2 * p*batch);
	hipMalloc((void**)&dev_order, sizeof(unsigned int) * 2 * p*batch);

	dim3 dimGrid12(2 * p*q, batch, 1);
	// generateDev_A<<<dimGrid12, 128>>>(dev_A0, dev_A, dev_V, height0, width0, height, width, p, q);
	hipLaunchKernelGGL(generateDev_A, dimGrid12, 128, 0, 0, dev_A0, dev_A, dev_V, height0, width0, height, width, p, q);
	hipDeviceSynchronize();

	double *host_Fnorm, *dev_tempFnorm, *dev_Fnorm;
	hipMalloc((void**)&dev_tempFnorm, sizeof(double) * 2 * p*batch);
	hipMalloc((void**)&dev_Fnorm, sizeof(double) * batch);
	host_Fnorm = (double*)malloc(sizeof(double) *batch);
	// computeFnorm1<<<2 * p*batch, 128 >>>(dev_A, dev_tempFnorm, p, q, height, width);
	hipLaunchKernelGGL(computeFnorm1, 2 * p*batch, 128, 0, 0, dev_A, dev_tempFnorm, p, q, height, width);
	// computeFnorm2<<<batch, 32 >>>(dev_tempFnorm, dev_Fnorm, p);
	hipLaunchKernelGGL(computeFnorm2, batch, 32, 0, 0, dev_tempFnorm, dev_Fnorm, p);

	dim3 dimGrid0(1, 1, 1);
	dim3 dimBlock0(k, k, 1);
	// generate_roundRobin_64<<<dimGrid0, dimBlock0>>>(dev_roundRobin, k);
	hipLaunchKernelGGL(generate_roundRobin_64, dimGrid0, dimBlock0, 0, 0, dev_roundRobin, k);

	// getRankNewNew<<<1, 1024>>>(2 * p);
	hipLaunchKernelGGL(getRankNewNew, 1, 1024, 0, 0, 2 * p);
	hipDeviceSynchronize();

	parallel_ordering_choice(p, q, dev_order, dev_A, height, width, dev_norm, batch);
	memset(host_allpass, 0, sizeof(unsigned)*batch);
	int sweep = 0;

	// steady_clock::time_point t3 = steady_clock::now();
	while (!ifallpass(host_allpass, batch, p) && sweep < 20) {
		for (int i = 0; i < 2 * p - 1; i++) {
			dim3 dimGrid77(sliceNum, p, batch);
			// generate_jointG00<<< dimGrid77, 256>>>(dev_A, height, width, dev_order, dev_pass, p, q, dev_pairsOfEVD, dev_AiAi, dev_AiAj, dev_AjAj, i, sliceNum);
			hipLaunchKernelGGL(generate_jointG00, dimGrid77, 256, 0, 0, dev_A, height, width, dev_order, dev_pass, p, q, dev_pairsOfEVD, dev_AiAi, dev_AiAj, dev_AjAj, i, sliceNum);
			hipDeviceSynchronize();

			dim3 dimGrid7(p, batch, 1);
			// generate_jointG21<<< dimGrid7, 256>>>(dev_jointG, dev_AiAi, dev_AiAj, dev_AjAj, dev_Fnorm, dev_pass, p, sliceNum);
			hipLaunchKernelGGL(generate_jointG21, dimGrid7, 256, 0, 0, dev_jointG, dev_AiAi, dev_AiAj, dev_AjAj, dev_Fnorm, dev_pass, p, sliceNum);
			hipDeviceSynchronize();

			EVD_(dev_jointG, dev_A, dev_V, dev_pairsOfEVD, p, q, height, width, dev_roundRobin, batch, sliceNum);
			hipDeviceSynchronize();
		}

		parallel_ordering_choice(p, q, dev_order, dev_A, height, width, dev_norm, batch);

		// judgeFunc<<< batch, 1024>>>(dev_allpass, dev_pass, p);
		hipLaunchKernelGGL(judgeFunc, batch, 1024, 0, 0, dev_allpass, dev_pass, p);
		hipDeviceSynchronize();
		hipMemcpy(host_allpass, dev_allpass, sizeof(unsigned)*batch, hipMemcpyDeviceToHost);
		hipDeviceSynchronize();
		// printf("sweep %d :completed.\n", sweep);
		sweep++;

	}
	hipDeviceSynchronize();

	dim3 dimGrid10(2 * p, batch, 1);
	dim3 dimBlock10(32, k, 1);
	// getUDV<<<dimGrid10, dimBlock10>>>(dev_A, dev_U, dev_V, dev_V0, height, width, height0, width0, p, q, dev_diag);
	hipLaunchKernelGGL(getUDV, dimGrid10, dimBlock10, 0, 0, dev_A, dev_U, dev_V, dev_V0, height, width, height0, width0, p, q, dev_diag);
	hipDeviceSynchronize();

	if (tag == 1) {
		int t = height0;
		height0 = width0;
		width0 = t;
		hipMalloc((void**)&dev_V, sizeof(double)* width0 * width0*batch);
		hipMemcpy(dev_V, dev_U, sizeof(double)* width0*width0*batch, hipMemcpyDeviceToDevice);
		hipMemcpy(dev_U, dev_V0, sizeof(double)*height0*height0*batch, hipMemcpyDeviceToDevice);
	}
	hipDeviceSynchronize();

	steady_clock::time_point t2 = steady_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

	// printf("matrix:%dx%dx%d, sweeps:%d, Total time: %lf(s), time3:%lf(s)\n", batch, height, width, sweep, time_span.count());
	
    // hipMemcpy(host_diag, dev_diag, sizeof(double) * height0, hipMemcpyDeviceToHost);
    // hipMemcpy(host_U, dev_U, sizeof(double) * height0*height0, hipMemcpyDeviceToHost);
    // hipMemcpy(host_V, dev_V0, sizeof(double) * width0*width0, hipMemcpyDeviceToHost);

    free(host_allpass);
    free(host_pass);
    free(host_Fnorm);

    hipDeviceReset();
	return time_span.count();
}

int main(){
	int n=100, h=512, w=512;
	// string file_path = "/public/home/ictapp_x/pyf_folder/comtest/svd_test18/data_in/input.txt";
	string file_path = "input.txt";

	FILE* A_fp = fopen(file_path.data(), "r");
	double* host_A = (double*)malloc(sizeof(double)*h*w);
    if(A_fp==NULL){
        printf("open file: %s failed!\n", file_path.data());
        return 0;
    }
    for(int i=0; i < h*w; i++){
        fscanf(A_fp, "%lf", &host_A[i]);
    }

	double t1 = magma_svd(file_path.data(), n, h, w);
	int shape[3] = {n, h, w};
	double t2 = svd_large_matrix(host_A, shape);

	printf("speedup: %lf / %lf = %lf\n", t1, t2, t1/t2);
}
