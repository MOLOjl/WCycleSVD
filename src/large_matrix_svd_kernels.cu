#define PRECISION 1e-10
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <time.h>
#include <float.h>

#include "result_print.cu"

using namespace std;

__device__ int p_a[2][2000];
__device__ int p1[2000];
__device__ int p_ab[2][2000][2000];
__device__ int pp[2000];
static double* gm_V = NULL;

// <<<(2*p*q, batch, 1), 128>>>
__global__ void generateDev_A(double *dev_A0, double *dev_A, double *dev_V, int height0, int width0, int height, int width, int p, int k, int q)
{
	unsigned bidxx = blockIdx.x % q;	// 0~7 % 2
	unsigned bidxy = blockIdx.x / q;	// 0~7 / 2
	unsigned tidx = threadIdx.x % 32;	// 0~31
	unsigned tidy = threadIdx.x / 32;	// 0~3
	int iter = k / 4;
	for (int i = 0; i < iter; i++)
	{

		if (((bidxy * k + 4 * i + tidy) < width0) && ((bidxx * 32 + tidx) < height0))
		{
			dev_A[blockIdx.y * height * width + ((bidxy * k + 4 * i + tidy) * height + bidxx * 32 + tidx)] = dev_A0[blockIdx.y * height0 * width0 + ((bidxy * k + 4 * i + tidy) * height0 + bidxx * 32 + tidx)];
		}
		else
		{
			dev_A[blockIdx.y * height * width + ((bidxy * k + 4 * i + tidy) * height + bidxx * 32 + tidx)] = 0;
		}
		if (bidxx == 0 && tidx == 0 && (bidxy * k + 4 * i + tidy) < width0)
		{
			dev_V[blockIdx.y * width * width + (bidxy * k + 4 * i + tidy) * width + bidxy * k + 4 * i + tidy] = 1;
		}	
	}
}

//<< <2 * p*batch, 128 >> >
__global__ void computeFnorm1(double *dev_A, double *temp_Fnorm, int p, int q, int height, int width, int k)
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
		tmp += __shfl_xor_sync(-1, tmp, s);
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
		tmp += __shfl_xor_sync(-1, tmp, s);
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
__global__ void compute_norm(double *dev_A, double *dev_norm, unsigned int *dev_order, int height, int width, int p, int q, int k)
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
		tmp += __shfl_xor_sync(-1, tmp, s);
	}
	if (locx == 0)
		sm_norm[locy] = tmp;
	__syncthreads();
	if (tid == 0)
	{
		dev_norm[bid] = sm_norm[0] + sm_norm[1] + sm_norm[2] + sm_norm[3];
	}
}

//<< <batch, 1024 >> > (dev_norm, dev_order, 2 * p, p);
__global__ void binoticSort_original(double *value, unsigned int *arr1, unsigned int len, int p)
{
	__shared__ double buf[1024];
	__shared__ unsigned int buf_index1[1024];
	int bid = blockIdx.x;
	// buf[threadIdx.x] = (threadIdx.x < len ? value[bid * 2 * p + threadIdx.x] : 0xffffffffu * 1.0);	// seriously?
	buf[threadIdx.x] = (threadIdx.x < len ? value[bid * 2 * p + threadIdx.x] : DBL_MAX);
	buf_index1[threadIdx.x] = (threadIdx.x < len ? arr1[bid * 2 * p + threadIdx.x] : 0);
	__syncthreads();

	for (unsigned kk = 2; kk <= blockDim.x; kk *= 2)
	{ //buid k elements ascend or descend
		for (unsigned jj = kk >> 1; jj > 0; jj >>= 1)
		{ //merge longer binotic into shorter binotic
			unsigned swapIdx = threadIdx.x ^ jj;	// 按位异或
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

	// if(threadIdx.x==0)
	// {
	// 	printf("order:%d-%d-%d-%d, norm:%lf-%lf-%lf-%lf\n", arr1[0], arr1[1], arr1[2], arr1[3], 
	// 		value[0], value[1], value[2], value[3]);
	// }
}

void parallel_ordering_choice(int p, int q, unsigned int *dev_order, double *dev_A, int height, int width, double *dev_norm, int batch, int k)
{

	compute_norm<<<2 * p * batch, 128>>>(dev_A, dev_norm, dev_order, height, width, p, q, k);
	binoticSort_original<<<batch, 1024>>>(dev_norm, dev_order, 2 * p, p);
	cudaDeviceSynchronize();
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

// <<<batch, 256>>>
__global__ void init_dev_V(double* dev_V, int width){
	int tid = threadIdx.x;
	for(int i=tid; i<width; i += blockDim.x)
		dev_V[blockIdx.x*width*width + i*width + i] = 1;
}

// <<<(p, batch), 256>>> max w = 128 
__global__ void match_Aij(double *dev_A, int height, int width, int iterNum, unsigned int *dev_order, int *dev_pairsOfEVD, double *dev_Aij, int p, int k){
	__shared__ int index[2];	// 配对的列块索引i和j
	__shared__ double temp[256];
	int tid = threadIdx.x;		// 0~255
	if(tid==0 && k>128)	printf("error!!!\n");
	
	__syncthreads();

	if (tid == blockDim.x-1)
	{
		index[0] = dev_order[(2 * p - 1) - p_ab[0][blockIdx.x][iterNum] + blockIdx.y * 2 * p];
		index[1] = dev_order[(2 * p - 1) - p_ab[1][blockIdx.x][iterNum] + blockIdx.y * 2 * p];

		// this is used when updating A and V
		dev_pairsOfEVD[2 * (blockIdx.y * p + blockIdx.x)] = index[0];
		dev_pairsOfEVD[2 * (blockIdx.y * p + blockIdx.x) + 1] = index[1];

		// if(blockIdx.x==0 && blockIdx.y == 0)
		// 	printf("echo.. index: %d,%d\n", index[0], index[1]);
	}
	
	int offset = 0;		// used when write to dev_Aij
	if(tid < 128)
		offset = (tid%128)*height; 	// A_i
	else
		offset = (tid%128 + k)*height;
	
	__syncthreads();

	for(int step=0; step<height; step ++){
		if(tid%128 < k){
			temp[tid] = dev_A[blockIdx.y * height*width + (index[tid/128]*k + tid%128)*height + step];	// get ai aj
			dev_Aij[blockIdx.y * p*2*k*height + blockIdx.x * 2*k*height + offset] = temp[tid];	// store ai aj
		}

		offset ++;

		// if((index[tid/128]*k + tid%128)*height + step == 1)
		// 	printf("hello\n");
	}
}

//  <<<(p, batch), 16*16>>> max k = 128 min k = 16
__global__ void converge_verify(double* dev_G, int size, double *dev_Fnorm, unsigned int *dev_pass, int p, int k, int iterNum){
	int tid = threadIdx.x;
	__shared__ double aiaj[16][16];
	aiaj[tid/16][tid%16] = 0;	// init aiaj

	int idx = tid % 16 + k;
	int idy = tid / 16;
	double temp = 0;
	while(idx < 2*k){
		while(idy < k){
			temp = dev_G[(blockIdx.y*p + blockIdx.x) * 2*k*2*k + idx * 2*k + idy];
			aiaj[tid / 16][tid % 16] +=  temp * temp;
			idy += 16;
		}
		idx += 16;
		idy = tid / 16;
	}

	__syncthreads();

	if(tid/16 == 0){
		for(int i=1; i<16; i++)
			aiaj[tid/16][0] += aiaj[tid/16][i];
	}

	__syncthreads();

	if(tid == 0){
		for(int i=1; i<16; i++)
			aiaj[0][0] += aiaj[i][0];
		
		if(sqrt(aiaj[0][0]) > 10.0 * PRECISION * dev_Fnorm[blockIdx.y])
		{
			//not orthogonal
			dev_pass[p * blockIdx.y + blockIdx.x] = 0;
		}

		// if(blockIdx.x==0 && blockIdx.y==0 && iterNum==0){
		// 	printf("sqrt aiaj:%lf\n", sqrt(aiaj[0][0]));
		// }
	}
		
}

// generate Gram Matrix step1, match A_ij do the GEMM 
// <<<(sliceNum, p, batch), 256>>>
__global__ void generate_jointG00(double *dev_A, int height, int width, unsigned int *dev_order, unsigned int *dev_pass, int p, int q, int *dev_pairsOfEVD,
								  double *dev_AiAi, double *dev_AiAj, double *dev_AjAj, int iterNum, int k, int slice, int sliceNum)
{
	__shared__ int index[2];
	__shared__ double sm_Ai[32][25];
	__shared__ double sm_Aj[32][25];
	int tid = threadIdx.x;			// 0~255
	int tid1 = threadIdx.x + 256;	// 256~511
	int tid2 = threadIdx.x + 512;	// 512~767
	int iter = slice / 32;	// 1

	if (tid == 0)
	{
		index[0] = dev_order[(2 * p - 1) - p_ab[0][blockIdx.y][iterNum] + blockIdx.z * 2 * p];
		index[1] = dev_order[(2 * p - 1) - p_ab[1][blockIdx.y][iterNum] + blockIdx.z * 2 * p];
		dev_pairsOfEVD[2 * (blockIdx.z * p + blockIdx.y)] = index[0];
		dev_pairsOfEVD[2 * (blockIdx.z * p + blockIdx.y) + 1] = index[1];
		
		// if(blockIdx.x==1 && blockIdx.y==1)
		// printf("index: %d - %d \n", index[0], index[1]);
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
	int locx = tid % 32; 	// (0~31)
	int locy = tid / 32;	// (0~7)

	for (int t = 0; t < iter; t++)
	{
		if (true)
		{
			// get data
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

			// do GEMM
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
			// __syncthreads();
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
			// __syncthreads();
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
	
	// store data
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

// generate Gram Matrix step2, fill jointG and do the converge verification
// <<<(p, batch, 1), 256>>>
__global__ void generate_jointG21(double *dev_jointG, double *dev_AiAi, double *dev_AiAj, double *dev_AjAj, double *dev_Fnorm, unsigned int *dev_pass, int p, int k, int sliceNum)
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
		locx = tid % k;	// 0~15
		locy = tid / k;	// 0~15

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
			d += __shfl_xor_sync(-1, d, s);
		}

		// 
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
			if (sqrt(sum[0]) > 100.0 * PRECISION * dev_Fnorm[blockIdx.y] * dev_Fnorm[blockIdx.y])
			{
				dev_pass[p * blockIdx.y + blockIdx.x] = 0;	// not converge
			}

		}
		__syncthreads();
        // fill G
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
				d += __shfl_xor_sync(-1, d, s);
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
			if (sqrt(sum[0]) > 10.0 * PRECISION * dev_Fnorm[blockIdx.y])
			{
				dev_pass[p * blockIdx.y + blockIdx.x] = 0;
			}
		}
		__syncthreads();
	}
}

// k=1
__global__ void updateBlockColumn2(double* dev_A, double* dev_V, double* dev_jointG, int* dev_pairsOfEVD, int p, int q, int height, int width, int k, int slice) {
	__shared__ double sm_A[32 * 48];
	__shared__ double sm_V[32 * 48];
	__shared__ double sm_G[48 * 48];   ///32* 2k    2k*2k  =  32*2k
	__shared__ unsigned index[2];
	int iter = slice / 32;
	int tid = threadIdx.x;

	if (tid == 0) {
		index[0] = dev_pairsOfEVD[2 * blockIdx.y];
		index[1] = dev_pairsOfEVD[2 * blockIdx.y + 1];
	}
	int locx, locy;
	for (int i = tid; i < 2 * k * 2 * k; i += 256) {
		locx = i % (2 * k);
		locy = i / (2 * k);
		sm_G[locx * 2 * k + locy] = dev_jointG[blockIdx.y * 2 * k * 2 * k + i];
	}
	__syncthreads();

	/////update U
	for (int t = 0;t < iter;t++) {
		if (true) {
			if (tid < 128) {
				for (int i = tid;i < 32 * k;i += 128) {
					locx = i % 32;
					locy = i / 32;
					sm_A[i] = dev_A[(blockIdx.y / p)*height*width + (index[0] * k + locy) * height + blockIdx.x*slice + t * 32 + locx];
				}
			}
			else {
				for (int i = (tid - 128);i < 32 * k;i += 128) {
					locx = i % 32;
					locy = i / 32;
					sm_A[32 * k + i] = dev_A[(blockIdx.y / p)*height*width + (index[1] * k + locy) * height + blockIdx.x*slice + t * 32 + locx];
				}
			}
			__syncthreads();
			if (tid < 128) {
				for (int i = tid;i < 32 * k;i += 128) {
					locx = i % 32;
					locy = i / 32;
					double tmp = 0;
					for (int j = 0;j < 2 * k;j++) {
						tmp += sm_A[j * 32 + locx] * sm_G[j * 2 * k + locy];/////////////////////////
					}
					dev_A[(blockIdx.y / p)*height*width + (index[0] * k + locy) * height + blockIdx.x*slice + t * 32 + locx] = tmp;
				}
			}
			else {
				for (int i = (tid - 128);i < 32 * k;i += 128) {
					locx = i % 32;
					locy = i / 32;
					double tmp = 0;
					for (int j = 0;j < 2 * k;j++) {
						tmp += sm_A[j * 32 + locx] * sm_G[j * 2 * k + locy + k];
					}
					dev_A[(blockIdx.y / p)*height*width + (index[1] * k + locy) * height + blockIdx.x*slice + t * 32 + locx] = tmp;
				}
			}
		}
		__syncthreads();
	}

	/////update V
	for (int t = 0;t < iter;t++) {
		if (tid < 128) {
			for (int i = tid;i < 32 * k;i += 128) {
				locx = i % 32;
				locy = i / 32;
				if ((blockIdx.x*slice + t * 32 + locx) < width) {
					sm_V[i] = dev_V[(blockIdx.y / p)*width*width + (index[0] * k + locy) * width + blockIdx.x*slice + t * 32 + locx];
				}
				else
					sm_V[i] = 0;
			}
		}
		else {
			for (int i = (tid - 128);i < 32 * k;i += 128) {
				locx = i % 32;
				locy = i / 32;
				if ((t * 32 + locx) < width) {
					sm_V[32 * k + i] = dev_V[(blockIdx.y / p)*width*width + (index[1] * k + locy) * width + blockIdx.x*slice + t * 32 + locx];
				}
				else
					sm_V[32 * k + i] = 0;
			}
		}
		__syncthreads();
		if (tid < 128) {
			for (int i = tid;i < 32 * k;i += 128) {
				locx = i % 32;
				locy = i / 32;
				double tmp = 0;
				for (int j = 0;j < 2 * k;j++) {
					tmp += sm_V[j * 32 + locx] * sm_G[j * 2 * k + locy];
				}
				if ((t * 32 + locx) < width) {
					dev_V[(blockIdx.y / p)*width*width + (index[0] * k + locy) * width + blockIdx.x*slice + t * 32 + locx] = tmp;
				}
			}
		}
		else {
			for (int i = (tid - 128);i < 32 * k;i += 128) {
				locx = i % 32;
				locy = i / 32;
				double tmp = 0;
				for (int j = 0;j < 2 * k;j++) {
					tmp += sm_V[j * 32 + locx] * sm_G[j * 2 * k + locy + k];
				}
				if ((t * 32 + locx) < width) {
					dev_V[(blockIdx.y / p)*width*width + (index[1] * k + locy) * width + blockIdx.x*slice + t * 32 + locx] = tmp;
				}
			}

		}
		__syncthreads();
	}

}

// <<< (sliceNum, p, batch), 128 >>>
__global__ void updateBlockColumn2_4(double *dev_A, double *dev_V, double *dev_jointG, int *dev_pairsOfEVD, int p, int q, int height, int width, int k, int slice)
{
	__shared__ double sm_A[128 * 2];
	__shared__ double sm_V[128 * 2];
	__shared__ double sm_G[8][8];
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
	for (int i = tid; i < 64; i += 128)
	{
		sm_G[i / (2 * k)][i % (2 * k)] = dev_jointG[(blockIdx.z * p + blockIdx.y) * 64 + i];
	}
	__syncthreads();

	double Avalue1 = 0.0;
	double Avalue2 = 0.0;
	double Ivalue1 = 0.0;
	double Ivalue2 = 0.0;

	for (int t = 0; t < iter; t++)
	{
		if (true)
		{
			sm_A[tid] = dev_A[blockIdx.z * height * width + (index[0] * k + locy) * height + blockIdx.x * slice + t * 32 + locx];
			sm_A[tid + 128] = dev_A[blockIdx.z * height * width + (index[1] * k + locy) * height + blockIdx.x * slice + t * 32 + locx];
			if ((blockIdx.x * slice + t * 32 + locx) < width)
			{
				sm_V[tid] = dev_V[blockIdx.z * width * width + (index[0] * k + locy) * width + blockIdx.x * slice + t * 32 + locx];
				sm_V[tid + 128] = dev_V[blockIdx.z * width * width + (index[1] * k + locy) * width + blockIdx.x * slice + t * 32 + locx];
			}
			else
			{
				sm_V[tid] = 0;
				sm_V[tid + 128] = 0;
			}
			__syncthreads();

			Avalue1 = 0.0;
			Avalue2 = 0.0;
			Ivalue1 = 0.0;
			Ivalue2 = 0.0;

			for (unsigned j = 0; j < 2 * k; j++)
			{
				Avalue1 += sm_A[locx + 32 * j] * sm_G[locy][j];
				Avalue2 += sm_A[locx + 32 * j] * sm_G[locy + 4][j];
			}
			if ((blockIdx.x * slice + t * 32 + locx) < width)
			{
				for (unsigned j = 0; j < 2 * k; j++)
				{
					Ivalue1 += sm_V[locx + 32 * j] * sm_G[locy][j];
					Ivalue2 += sm_V[locx + 32 * j] * sm_G[locy + 4][j];
				}
			}
			__syncthreads();

			dev_A[blockIdx.z * height * width + (index[0] * k + locy) * height + blockIdx.x * slice + t * 32 + locx] = Avalue1;
			dev_A[blockIdx.z * height * width + (index[1] * k + locy) * height + blockIdx.x * slice + t * 32 + locx] = Avalue2;
			if ((blockIdx.x * slice + t * 32 + locx) < width)
			{
				dev_V[blockIdx.z * width * width + (index[0] * k + locy) * width + blockIdx.x * slice + t * 32 + locx] = Ivalue1;
				dev_V[blockIdx.z * width * width + (index[1] * k + locy) * width + blockIdx.x * slice + t * 32 + locx] = Ivalue2;
			}
			__syncthreads();
		}
		__syncthreads();
	}
}

// <<<(sliceNum, p, batch), 128>>>
__global__ void updateBlockColumn2_8(double *dev_A, double *dev_V, double *dev_jointG, int *dev_pairsOfEVD, int p, int q, int height, int width, int k, int slice)
{
	__shared__ double sm_A[32 * 8 * 2];
	__shared__ double sm_V[32 * 8 * 2];
	__shared__ double sm_G[16][16];
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
	for (int i = tid; i < 256; i += 128)
	{
		sm_G[i / (2 * k)][i % (2 * k)] = dev_jointG[blockIdx.y * 2 * k * 2 * k + i];
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
		if (true)
		{
			sm_A[tid] = dev_A[blockIdx.z * height * width + (index[0] * k + locy) * height + blockIdx.x * slice + t * 32 + locx];
			sm_A[tid + 128] = dev_A[blockIdx.z * height * width + (index[0] * k + locy + 4) * height + blockIdx.x * slice + t * 32 + locx];
			sm_A[tid + 256] = dev_A[blockIdx.z * height * width + (index[1] * k + locy) * height + blockIdx.x * slice + t * 32 + locx];
			sm_A[tid + 384] = dev_A[blockIdx.z * height * width + (index[1] * k + locy + 4) * height + blockIdx.x * slice + t * 32 + locx];
			if ((blockIdx.x * slice + t * 32 + locx) < width)
			{
				sm_V[tid] = dev_V[blockIdx.z * width * width + (index[0] * k + locy) * width + blockIdx.x * slice + t * 32 + locx];
				sm_V[tid + 128] = dev_V[blockIdx.z * width * width + (index[0] * k + locy + 4) * width + blockIdx.x * slice + t * 32 + locx];
				sm_V[tid + 256] = dev_V[blockIdx.z * width * width + (index[1] * k + locy) * width + blockIdx.x * slice + t * 32 + locx];
				sm_V[tid + 384] = dev_V[blockIdx.z * width * width + (index[1] * k + locy + 4) * width + blockIdx.x * slice + t * 32 + locx];
			}
			else
			{
				sm_V[tid] = 0;
				sm_V[tid + 128] = 0;
				sm_V[tid + 256] = 0;
				sm_V[tid + 384] = 0;
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
				Avalue1 += sm_A[locx + 32 * j] * sm_G[tid / 32][j];
				Avalue11 += sm_A[locx + 32 * j] * sm_G[tid / 32 + 4][j];
				Avalue2 += sm_A[locx + 32 * j] * sm_G[tid / 32 + 8][j];
				Avalue22 += sm_A[locx + 32 * j] * sm_G[tid / 32 + 12][j];
			}
			if ((blockIdx.x * slice + t * 32 + locx) < width)
			{
				for (unsigned j = 0; j < 2 * k; j++)
				{
					Ivalue1 += sm_V[locx + 32 * j] * sm_G[tid / 32][j];
					Ivalue11 += sm_V[locx + 32 * j] * sm_G[tid / 32 + 4][j];
					Ivalue2 += sm_V[locx + 32 * j] * sm_G[tid / 32 + 8][j];
					Ivalue22 += sm_V[locx + 32 * j] * sm_G[tid / 32 + 12][j];
				}
			}
			__syncthreads();
			
			dev_A[blockIdx.z * height * width + (index[0] * k + locy) * height + blockIdx.x * slice + t * 32 + locx] = Avalue1;
			dev_A[blockIdx.z * height * width + (index[0] * k + locy + 4) * height + blockIdx.x * slice + t * 32 + locx] = Avalue11;
			dev_A[blockIdx.z * height * width + (index[1] * k + locy) * height + blockIdx.x * slice + t * 32 + locx] = Avalue2;
			dev_A[blockIdx.z * height * width + (index[1] * k + locy + 4) * height + blockIdx.x * slice + t * 32 + locx] = Avalue22;
			if ((blockIdx.x * slice + t * 32 + locx) < width)
			{
				dev_V[blockIdx.z * width * width + (index[0] * k + locy) * width + blockIdx.x * slice + t * 32 + locx] = Ivalue1;
				dev_V[blockIdx.z * width * width + (index[0] * k + locy + 4) * width + blockIdx.x * slice + t * 32 + locx] = Ivalue11;
				dev_V[blockIdx.z * width * width + (index[1] * k + locy) * width + blockIdx.x * slice + t * 32 + locx] = Ivalue2;
				dev_V[blockIdx.z * width * width + (index[1] * k + locy + 4) * width + blockIdx.x * slice + t * 32 + locx] = Ivalue22;
			}
			__syncthreads();
		}
		__syncthreads();
	}
}

// <<< (sliceNum, p, batch), 256 >>>
__global__ void updateBlockColumn2_16(double *dev_A, double *dev_V, double *dev_jointG, int *dev_pairsOfEVD, int p, int q, int height, int width, int k, int slice)
{
	__shared__ double sm_A[32 * 16 * 2];	// 1024
	__shared__ double sm_V[32 * 16 * 2];	// 1024
	__shared__ double sm_G[32][32];			// 1024
	__shared__ unsigned index[2];
	int iter = slice / 32;
	int tid = threadIdx.x;	// 0~255
	int locx, locy;
	locx = tid % 32;	// (0~31)
	locy = tid / 32;

	if (tid < 2)
	{
		index[tid] = dev_pairsOfEVD[2 * (blockIdx.z * p + blockIdx.y) + tid];
	}
	for (int i = tid; i < 1024; i += 256)
	{
		sm_G[i / 32][locx] = dev_jointG[(blockIdx.z * p + blockIdx.y) * 1024 + i];		// 修改后
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
		if (true)
		{
			sm_A[tid] = dev_A[blockIdx.z * height*width + (index[0] * k + locy) * height + blockIdx.x * slice + t * 32 + locx];
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
				for (unsigned j = 0; j < 2 * k; j++)
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

// <<< (sliceNum, p, batch), 256 >>>
__global__ void updateBlockColumn2_24(double *dev_A, double *dev_V, double *dev_jointG, int *dev_pairsOfEVD, int p, int q, int height, int width, int k, int slice)
{
	__shared__ double sm_A[32 * 24 * 2];
	__shared__ double sm_V[32 * 24 * 2];
	__shared__ double sm_G[48][48];
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
	for (int i = tid; i < 2 * k * 2 * k; i += 256)
	{
		sm_G[i / (2 * k)][i % (2 * k)] = dev_jointG[(blockIdx.z * p + blockIdx.y) * 2304 + i];
	}
	__syncthreads();

	double Avalue11 = 0.0;
	double Avalue12 = 0.0;
	double Avalue13 = 0.0;
	double Avalue21 = 0.0;
	double Avalue22 = 0.0;
	double Avalue23 = 0.0;
	double Ivalue11 = 0.0;
	double Ivalue12 = 0.0;
	double Ivalue13 = 0.0;
	double Ivalue21 = 0.0;
	double Ivalue22 = 0.0;
	double Ivalue23 = 0.0;

	for (int t = 0; t < iter; t++)
	{

		if (true)
		{

			sm_A[tid] = dev_A[blockIdx.z * height * width + (index[0] * k + locy) * height + blockIdx.x * slice + t * 32 + locx];
			sm_A[tid + 256] = dev_A[blockIdx.z * height * width + (index[0] * k + locy + 8) * height + blockIdx.x * slice + t * 32 + locx];
			sm_A[tid + 512] = dev_A[blockIdx.z * height * width + (index[0] * k + locy + 16) * height + blockIdx.x * slice + t * 32 + locx];
			sm_A[tid + 768] = dev_A[blockIdx.z * height * width + (index[1] * k + locy) * height + blockIdx.x * slice + t * 32 + locx];
			sm_A[tid + 1024] = dev_A[blockIdx.z * height * width + (index[1] * k + locy + 8) * height + blockIdx.x * slice + t * 32 + locx];
			sm_A[tid + 1280] = dev_A[blockIdx.z * height * width + (index[1] * k + locy + 16) * height + blockIdx.x * slice + t * 32 + locx];

			if ((blockIdx.x * slice + t * 32 + locx) < width)
			{
				sm_V[tid] = dev_V[blockIdx.z * width * width + (index[0] * k + locy) * width + blockIdx.x * slice + t * 32 + locx];
				sm_V[tid + 256] = dev_V[blockIdx.z * width * width + (index[0] * k + locy + 8) * width + blockIdx.x * slice + t * 32 + locx];
				sm_V[tid + 512] = dev_V[blockIdx.z * width * width + (index[0] * k + locy + 16) * width + blockIdx.x * slice + t * 32 + locx];
				sm_V[tid + 768] = dev_V[blockIdx.z * width * width + (index[1] * k + locy) * width + blockIdx.x * slice + t * 32 + locx];
				sm_V[tid + 1024] = dev_V[blockIdx.z * width * width + (index[1] * k + locy + 8) * width + blockIdx.x * slice + t * 32 + locx];
				sm_V[tid + 1280] = dev_V[blockIdx.z * width * width + (index[1] * k + locy + 16) * width + blockIdx.x * slice + t * 32 + locx];
			}
			else
			{
				sm_V[tid] = 0;
				sm_V[tid + 256] = 0;
				sm_V[tid + 512] = 0;
				sm_V[tid + 768] = 0;
				sm_V[tid + 1024] = 0;
				sm_V[tid + 1280] = 0;
			}

			__syncthreads();
			Avalue11 = 0.0;
			Avalue12 = 0.0;
			Avalue13 = 0.0;
			Avalue21 = 0.0;
			Avalue22 = 0.0;
			Avalue23 = 0.0;
			Ivalue11 = 0.0;
			Ivalue12 = 0.0;
			Ivalue13 = 0.0;
			Ivalue21 = 0.0;
			Ivalue22 = 0.0;
			Ivalue23 = 0.0;
			for (unsigned j = 0; j < 2 * k; j++)
			{
				Avalue11 += sm_A[locx + 32 * j] * sm_G[locy][j];
				Avalue12 += sm_A[locx + 32 * j] * sm_G[locy + 8][j];
				Avalue13 += sm_A[locx + 32 * j] * sm_G[locy + 16][j];
				Avalue21 += sm_A[locx + 32 * j] * sm_G[locy + 24][j];
				Avalue22 += sm_A[locx + 32 * j] * sm_G[locy + 32][j];
				Avalue23 += sm_A[locx + 32 * j] * sm_G[locy + 40][j];
			}
			if ((blockIdx.x * slice + t * 32 + locx) < width)
			{
				for (unsigned j = 0; j < 2 * k; j++)
				{
					Ivalue11 += sm_V[locx + 32 * j] * sm_G[locy][j];
					Ivalue12 += sm_V[locx + 32 * j] * sm_G[locy + 8][j];
					Ivalue13 += sm_V[locx + 32 * j] * sm_G[locy + 16][j];
					Ivalue21 += sm_V[locx + 32 * j] * sm_G[locy + 24][j];
					Ivalue22 += sm_V[locx + 32 * j] * sm_G[locy + 32][j];
					Ivalue23 += sm_V[locx + 32 * j] * sm_G[locy + 40][j];
				}
			}
			__syncthreads();

			dev_A[blockIdx.z * height * width + (index[0] * k + locy) * height + blockIdx.x * slice + t * 32 + locx] = Avalue11;
			dev_A[blockIdx.z * height * width + (index[0] * k + locy + 8) * height + blockIdx.x * slice + t * 32 + locx] = Avalue12;
			dev_A[blockIdx.z * height * width + (index[0] * k + locy + 16) * height + blockIdx.x * slice + t * 32 + locx] = Avalue13;
			dev_A[blockIdx.z * height * width + (index[1] * k + locy) * height + blockIdx.x * slice + t * 32 + locx] = Avalue21;
			dev_A[blockIdx.z * height * width + (index[1] * k + locy + 8) * height + blockIdx.x * slice + t * 32 + locx] = Avalue22;
			dev_A[blockIdx.z * height * width + (index[1] * k + locy + 16) * height + blockIdx.x * slice + t * 32 + locx] = Avalue23;
			if ((blockIdx.x * slice + t * 32 + locx) < width)
			{
				dev_V[blockIdx.z * width * width + (index[0] * k + locy) * width + blockIdx.x * slice + t * 32 + locx] = Ivalue11;
				dev_V[blockIdx.z * width * width + (index[0] * k + locy + 8) * width + blockIdx.x * slice + t * 32 + locx] = Ivalue12;
				dev_V[blockIdx.z * width * width + (index[0] * k + locy + 16) * width + blockIdx.x * slice + t * 32 + locx] = Ivalue13;
				dev_V[blockIdx.z * width * width + (index[1] * k + locy) * width + blockIdx.x * slice + t * 32 + locx] = Ivalue21;
				dev_V[blockIdx.z * width * width + (index[1] * k + locy + 8) * width + blockIdx.x * slice + t * 32 + locx] = Ivalue22;
				dev_V[blockIdx.z * width * width + (index[1] * k + locy + 16) * width + blockIdx.x * slice + t * 32 + locx] = Ivalue23;
			}
			__syncthreads();
		}
		__syncthreads();
	}
}

// general update kernel
// <<<(p, batch), 64>>> min w=8, max w=128, and w has 8 as its factor (w/8=0)
__global__ void update_AV(double *dev_A, double *dev_V, double *dev_jointG, int *dev_pairsOfEVD, int p, int height, int width, int k){
	__shared__ double sm_A[8][128];
	__shared__ double sm_Jacobi[128][8];	// transformation of dev_jointG
	__shared__ unsigned index[2];

	int tid = threadIdx.x;		// 0~255
	int locx, locy, stride_y;
	locy = tid / k;				// (0-1)
	stride_y = blockDim.x / k;	// 2
	locx = tid % k;				// (0-31)
	
	if (tid < 2)
	{
		index[tid] = dev_pairsOfEVD[2 * (blockIdx.y * p + blockIdx.x) + tid];
	}

	__syncthreads();
	
	double result;
	for(int step = 0; step < height; step += 8){

		for(int i = locy; i < 8; i += stride_y){
			if(step+i < height){
				sm_A[i][locx] = dev_A[blockIdx.y * height*width + (index[0] * k + locx) * height + step+i];
				sm_A[i][locx + k] = dev_A[blockIdx.y * height*width + (index[1] * k + locx) * height + step+i];
			}
		}

		__syncthreads();

		for(int iter=0; iter<2*k; iter += 8){
			// load Jacobi matrix
			for(int i = locy; i < 8; i += stride_y){
				sm_Jacobi[locx][i] = dev_jointG[(blockIdx.y*p + blockIdx.x) * 2*k * 2*k + (iter+i) * 2*k + locx];
				sm_Jacobi[locx + k][i] = dev_jointG[(blockIdx.y*p + blockIdx.x) * 2*k * 2*k + (iter+i) * 2*k + locx+k];
			}
			
			__syncthreads();
			
			// do vector innner product
			result = 0;
			for(int i=0; i < 2*k; i++){
				result += sm_A[tid/8][i] * sm_Jacobi[i][tid%8];
			}
			
			if(iter + tid%8 < k)
				dev_A[blockIdx.y * height*width + (index[0]*k + iter+tid%8)*height + step+tid/8] = result;
			else
				dev_A[blockIdx.y * height*width + (index[1]*k + iter+tid%8-k)*height + step+tid/8] = result;
			
			__syncthreads();
			// TODO: update V
			if(iter + tid%8 < k)
				dev_V[blockIdx.y * height*width + (index[0]*k + iter+tid%8)*height + step+tid/8] = 0;
			else
				dev_V[blockIdx.y * height*width + (index[1]*k + iter+tid%8-k)*height + step+tid/8] = 0;
			
		}	
		__syncthreads();
	}

}

// k=1
__global__ void myevd_batched(double* dev_jointG, int* dev_roundRobin, int p, int k) {
	__shared__ double shared_G[32][32];
	__shared__ int shared_roundRobin[31][32];
	__shared__ int step;
	__shared__ double shared_V[32][32];
	__shared__ double shared_operators[2][32];
	__shared__ int shared_pairs[2][32];
	shared_G[threadIdx.x][threadIdx.y] = dev_jointG[blockIdx.y * 2 * k * 2 * k*p + blockIdx.x * 2 * k * 2 * k + threadIdx.y * 2 * k + threadIdx.x];

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

__global__ void myevd_batched_4(double *dev_jointG, int *dev_roundRobin, int p, int k)
{
	__shared__ double shared_G[8][8];
	__shared__ int shared_roundRobin[7][8];
	__shared__ int step;
	__shared__ double shared_V[8][8];
	__shared__ double shared_operators[2][8];
	__shared__ int shared_pairs[2][8];

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

__global__ void myevd_batched_8(double *dev_jointG, int *dev_roundRobin, int p, int k)
{
	__shared__ double shared_G[16][16];
	__shared__ int shared_roundRobin[15][16];
	__shared__ int step;
	__shared__ double shared_V[16][16];
	__shared__ double shared_operators[2][16];
	__shared__ int shared_pairs[2][16];
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

__global__ void myevd_batched_16(double *dev_jointG, int *dev_roundRobin, int p, int k)
{
	__shared__ double shared_G[32][32];
	__shared__ int shared_roundRobin[31][32];
	__shared__ double shared_V[32][32];
	__shared__ double shared_operators[2][32];
	__shared__ int shared_pairs[2][32];
	__shared__ int step;
	
	shared_G[threadIdx.y][threadIdx.x] = dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + threadIdx.x * 2 * k + threadIdx.y];

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

__global__ void muti_evd_batched_16(double *dev_jointG, int *dev_roundRobin, int p, int k)
{
	__shared__ double shared_G[32][32];
	__shared__ int shared_roundRobin[31][32];
	__shared__ double shared_V[32][32];
	__shared__ double shared_operators[2][32];
	__shared__ int shared_pairs[2][32];
	__shared__ int step;
	
	shared_G[threadIdx.y][threadIdx.x] = dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + threadIdx.x * 2 * k + threadIdx.y];

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
	int test = 10;
	while (test>0)
	{
		test --;
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
	}
	dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + threadIdx.y * 2 * k + threadIdx.x] = shared_V[threadIdx.x][threadIdx.y];
	__syncthreads();
}

__global__ void mysvd_batched_even(double *dev_A, int height0, int width0, double* dev_SigmaU, double *dev_V, int *dev_roundRobin)
{
	__shared__ double sm_A[48][48];
	__shared__ double sm_V[48][48];
	__shared__ int sm_roundRobin[47 * 48];
	__shared__ int sm_sign[24];
	__shared__ int ifConverged; 
	int bid = blockIdx.x;   // 0
	int tid = threadIdx.x;	// 0-31
	int laneid = threadIdx.x % 16;	// 0-15, 0-15
	int warpid = threadIdx.x / 16;  // 0-0, 1-1

	for (int i = tid; i < width0 * height0; i += blockDim.x)
	{
		sm_A[i / height0][i % height0] = dev_A[bid * height0 * width0 + i];
	}
	if (tid == 0)
		ifConverged = 1;
	
	for (int i = tid; i < 48*48; i += blockDim.x){
		sm_V[i/48][i%48] = (i/48 == i%48);
	}

	for (int i = tid; i < (width0 - 1) * width0; i += blockDim.x)
	{
		sm_roundRobin[i] = dev_roundRobin[i];
	}
	
	__syncthreads();

	double ai0 = 0, ai1 = 0, ai2 = 0;
	double aj0 = 0, aj1 = 0, aj2 = 0;
	double aiai_ajaj = 0;
	double aiaj = 0;
	double vi = 0;
	double vj = 0;
	double tan, cos, sin;
	int indexi = 0;
	int indexj = 0;
	int s = 8;
	int iter = 1;
	double tao;
	double signTao;
	int test = 1;
	// while (test > 0)
	while (ifConverged > 0)
	{
		iter = 1;
		test--;
		while (iter < width0)
		// while (iter < 2)
		{
			indexi = sm_roundRobin[(iter - 1) * width0 + warpid];
			indexj = sm_roundRobin[(iter - 1) * width0 + width0 - 1 - warpid];
			if(laneid < height0)
			{
				ai0 = sm_A[indexi][laneid];
				aj0 = sm_A[indexj][laneid];
			}
			else
			{
				ai0 = 0;
				aj0 = 0;				
			}
			if ((laneid + 16) < height0)
			{
				ai1 = sm_A[indexi][laneid + 16];
				aj1 = sm_A[indexj][laneid + 16];
			}
			else
			{
				ai1 = 0;
				aj1 = 0;
			}
			if ((laneid + 32) < height0)
			{
				ai2 = sm_A[indexi][laneid + 32];
				aj2 = sm_A[indexj][laneid + 32];
			}
			else
			{
				ai2 = 0;
				aj2 = 0;
			}

			aiai_ajaj = ai0 * ai0 - aj0 * aj0 + ai1 * ai1 - aj1 * aj1 + ai2 * ai2 - aj2 * aj2;
			aiaj = ai0 * aj0 + ai1 * aj1 + ai2 * aj2;

			if (warpid < ((width0 / 2 - 1) / 2 + 1) * 2)
			{
				for (s = 8; s > 0; s /= 2)
				{
					aiai_ajaj += __shfl_xor_sync(0xffffffff, aiai_ajaj, s);
					aiaj += __shfl_xor_sync(0xffffffff, aiaj, s);
				}
			}
			if (aiaj != 0)
			{
				
				tao = aiai_ajaj / (2 * aiaj);
				if (tao > 0)
					signTao = 1;
				if (tao == 0)
					signTao = 0;
				if (tao < 0)
					signTao = -1;
				tan = signTao / ((fabs(tao) + sqrt(1 + tao * tao)));
				cos = 1 / sqrt(1 + tan * tan);
				sin = tan * cos;
				//update A
				sm_A[indexi][laneid] = ai0 * cos + aj0 * sin;
				sm_A[indexj][laneid] = -ai0 * sin + aj0 * cos;
				if ((laneid + 16) < height0)
				{
					sm_A[indexi][laneid + 16] = ai1 * cos + aj1 * sin;
					sm_A[indexj][laneid + 16] = -ai1 * sin + aj1 * cos;
				}
				if ((laneid + 32) < height0)
				{
					sm_A[indexi][laneid + 32] = ai2 * cos + aj2 * sin;
					sm_A[indexj][laneid + 32] = -ai2 * sin + aj2 * cos;
				}

				//update V 
				if (laneid < width0)
				{
					vi = sm_V[indexi][laneid] * cos + sm_V[indexj][laneid] * sin;
					vj = -sm_V[indexi][laneid] * sin + sm_V[indexj][laneid] * cos;
					sm_V[indexi][laneid] = vi;
					sm_V[indexj][laneid] = vj;
				}
				if ((laneid + 16) < width0)
				{
					vi = sm_V[indexi][laneid + 16] * cos + sm_V[indexj][laneid + 16] * sin;
					vj = -sm_V[indexi][laneid + 16] * sin + sm_V[indexj][laneid + 16] * cos;
					sm_V[indexi][laneid + 16] = vi;
					sm_V[indexj][laneid + 16] = vj;
				}
				if ((laneid + 32) < width0)
				{
					vi = sm_V[indexi][laneid + 32] * cos + sm_V[indexj][laneid + 32] * sin;
					vj = -sm_V[indexi][laneid + 32] * sin + sm_V[indexj][laneid + 32] * cos;
					sm_V[indexi][laneid + 32] = vi;
					sm_V[indexj][laneid + 32] = vj;
				}
			}
			
			__syncthreads();
			iter++;			
		}

		if (laneid == 0)
		{
			if (fabs(aiaj) < PRECISION)
				sm_sign[warpid] = 0;
			else
				sm_sign[warpid] = 1;
		}
		__syncthreads();
		if (tid == 0)
		{
			ifConverged = 0;
			for (int i = 0; i < width0 / 2; i++)
			{
				ifConverged += sm_sign[i];
			}
		}
		__syncthreads();
	}
	
	__syncthreads();

	// get sigma * U
	double aii0 = 0, aii1 = 0, aii2 = 0;
	if (laneid < height0)
		aii0 = sm_A[warpid][laneid];
	if ((laneid + 16) < height0)
		aii1 = sm_A[warpid][laneid + 16];
	if ((laneid + 32) < height0)
		aii2 = sm_A[warpid][laneid + 32];

	if (laneid < height0)
		dev_SigmaU[bid * height0 * height0 + warpid * height0 + laneid] = aii0;
	if ((laneid + 16) < height0)
		dev_SigmaU[bid * height0 * height0 + warpid * height0 + laneid + 16] = aii1;
	if ((laneid + 32) < height0)
		dev_SigmaU[bid * height0 * height0 + warpid * height0 + laneid + 32] = aii2;

	if (laneid < height0)
		aii0 = sm_A[warpid + width0 / 2][laneid];
	if ((laneid + 16) < height0)
		aii1 = sm_A[warpid + width0 / 2][laneid + 16];
	if ((laneid + 32) < height0)
		aii2 = sm_A[warpid + width0 / 2][laneid + 32];

	if (laneid < height0)
		dev_SigmaU[bid * height0 * height0 + (warpid + width0 / 2) * height0 + laneid] = aii0;
	if ((laneid + 16) < height0)
		dev_SigmaU[bid * height0 * height0 + (warpid + width0 / 2) * height0 + laneid + 16] = aii1;
	if ((laneid + 32) < height0)
		dev_SigmaU[bid * height0 * height0 + (warpid + width0 / 2) * height0 + laneid + 32] = aii2;
	__syncthreads();
}

__global__ void mysvd_batched_even1(double *dev_A, int height0, int width0, double* dev_U, double *dev_V, int *dev_roundRobin)
{
	__shared__ double sm_A[48][48];
	__shared__ double sm_V[48][48];
	__shared__ int sm_roundRobin[47 * 48];
	__shared__ int sm_sign[24];
	__shared__ int ifConverged; 
	int bid = blockIdx.x;   // 0
	int tid = threadIdx.x;	// 0-31
	int laneid = threadIdx.x % 16;	// 0-15, 0-15
	int warpid = threadIdx.x / 16;  // 0-0, 1-1

	for (int i = tid; i < width0 * height0; i += blockDim.x)
	{
		sm_A[i / height0][i % height0] = dev_A[bid * height0 * width0 + i]; // 把A从gm取到sm中
	}
	if (tid == 0)
		ifConverged = 1;
	
	// 初始化V为单位矩阵
	for (int i = tid; i < 48*48; i += blockDim.x){
		sm_V[i/48][i%48] = (i/48 == i%48);
	}

	for (int i = tid; i < (width0 - 1) * width0; i += blockDim.x)
	{
		sm_roundRobin[i] = dev_roundRobin[i];	// 把环序列矩阵从gm取到sm中
	}
	__syncthreads();

	double ai0 = 0, ai1 = 0, ai2 = 0;
	double aj0 = 0, aj1 = 0, aj2 = 0;
	double aiai_ajaj = 0;
	double aiaj = 0;
	double vi = 0;
	double vj = 0;
	double tan, cos, sin;
	int indexi = 0;
	int indexj = 0;
	int s = 8;
	int iter = 1;
	double tao;
	double signTao;
	int test = 10;
	// while (test > 0)
	while (ifConverged > 0)
	{
		iter = 1;
		test--;
		while (iter < width0)
		{
			indexi = sm_roundRobin[(iter - 1) * width0 + warpid];	// [0-0] 0, [1-1] 2
			indexj = sm_roundRobin[(iter - 1) * width0 + width0 - 1 - warpid]; // [3-3] 1, ]2-2] 3
			if(laneid < height0)
			{
				ai0 = sm_A[indexi][laneid];
				aj0 = sm_A[indexj][laneid];
			}
			else
			{
				ai0 = 0;
				aj0 = 0;				
			}
			if ((laneid + 16) < height0)
			{
				ai1 = sm_A[indexi][laneid + 16];
				aj1 = sm_A[indexj][laneid + 16];
			}
			else
			{
				ai1 = 0;
				aj1 = 0;
			}
			if ((laneid + 32) < height0)
			{
				ai2 = sm_A[indexi][laneid + 32];
				aj2 = sm_A[indexj][laneid + 32];
			}
			else
			{
				ai2 = 0;
				aj2 = 0;
			}

			aiai_ajaj = ai0 * ai0 - aj0 * aj0 + ai1 * ai1 - aj1 * aj1 + ai2 * ai2 - aj2 * aj2;
			aiaj = ai0 * aj0 + ai1 * aj1 + ai2 * aj2;

			if (warpid < ((width0 / 2 - 1) / 2 + 1) * 2)
			{
				for (s = 8; s > 0; s /= 2)
				{
					aiai_ajaj += __shfl_xor_sync(0xffffffff, aiai_ajaj, s);
					aiaj += __shfl_xor_sync(0xffffffff, aiaj, s);
				}
			}
			if (aiaj != 0)
			{
				// if(tid==0)	printf("aiaj:%lf\n",aiaj);
				
				tao = aiai_ajaj / (2 * aiaj);
				if (tao > 0)
					signTao = 1;
				if (tao == 0)
					signTao = 0;
				if (tao < 0)
					signTao = -1;
				tan = signTao / ((fabs(tao) + sqrt(1 + tao * tao)));
				cos = 1 / sqrt(1 + tan * tan);
				sin = tan * cos;
				//update A
				sm_A[indexi][laneid] = ai0 * cos + aj0 * sin;
				sm_A[indexj][laneid] = -ai0 * sin + aj0 * cos;
				if ((laneid + 16) < height0)
				{
					sm_A[indexi][laneid + 16] = ai1 * cos + aj1 * sin;
					sm_A[indexj][laneid + 16] = -ai1 * sin + aj1 * cos;
				}
				if ((laneid + 32) < height0)
				{
					sm_A[indexi][laneid + 32] = ai2 * cos + aj2 * sin;
					sm_A[indexj][laneid + 32] = -ai2 * sin + aj2 * cos;
				}

				//update V ///////////////////////////////////////////////////////////////////////////
				if (laneid < width0)
				{
					vi = sm_V[indexi][laneid] * cos + sm_V[indexj][laneid] * sin;
					vj = -sm_V[indexi][laneid] * sin + sm_V[indexj][laneid] * cos;
					sm_V[indexi][laneid] = vi;
					sm_V[indexj][laneid] = vj;
				}
				if ((laneid + 16) < width0)
				{
					vi = sm_V[indexi][laneid + 16] * cos + sm_V[indexj][laneid + 16] * sin;
					vj = -sm_V[indexi][laneid + 16] * sin + sm_V[indexj][laneid + 16] * cos;
					sm_V[indexi][laneid + 16] = vi;
					sm_V[indexj][laneid + 16] = vj;
				}
				if ((laneid + 32) < width0)
				{
					vi = sm_V[indexi][laneid + 32] * cos + sm_V[indexj][laneid + 32] * sin;
					vj = -sm_V[indexi][laneid + 32] * sin + sm_V[indexj][laneid + 32] * cos;
					sm_V[indexi][laneid + 32] = vi;
					sm_V[indexj][laneid + 32] = vj;
				}
			}
			
			__syncthreads();
			iter++;			
		}
		// if(tid == 0) printf("_aiaj:%lf\n", aiaj);
		if (laneid == 0)
		{
			if (fabs(aiaj) < PRECISION)
				sm_sign[warpid] = 0;
			else
				sm_sign[warpid] = 1;
		}
		__syncthreads();
		if (tid == 0)
		{
			ifConverged = 0;
			for (int i = 0; i < width0 / 2; i++)
			{
				ifConverged += sm_sign[i];
			}
		}
		__syncthreads();
	}

	// if(tid==0 && blockIdx.x==0) printf("sweeps:%d\n", 10 - test);

	__syncthreads();

	// get V
	if (laneid < width0)
		dev_U[bid * width0 * width0 + warpid * width0 + laneid] = sm_V[warpid][laneid];
	if ((laneid + 16) < width0)
		dev_U[bid * width0 * width0 + warpid * width0 + laneid + 16] = sm_V[warpid][laneid + 16];
	if ((laneid + 32) < width0)
		dev_U[bid * width0 * width0 + warpid * width0 + laneid + 32] = sm_V[warpid][laneid + 32];
	__syncthreads();
	if (laneid < width0)
		dev_U[bid * width0 * width0 + (warpid + width0 / 2) * width0 + laneid] = sm_V[warpid + width0 / 2][laneid];
	if ((laneid + 16) < width0)
		dev_U[bid * width0 * width0 + (warpid + width0 / 2) * width0 + laneid + 16] = sm_V[warpid + width0 / 2][laneid + 16];
	if ((laneid + 32) < width0)
		dev_U[bid * width0 * width0 + (warpid + width0 / 2) * width0 + laneid + 32] = sm_V[warpid + width0 / 2][laneid + 32];
	
	__syncthreads();
}

// <<<batch, blockDim = min(size/4 * size/4, 1024)>>> max k=64 w=h
__global__ void mysvd_batched_even_gm_plus(double *dev_G, int size, double *gm_V, int *dev_roundRobin){
	__shared__ int sm_sign[64];	// size/2 <= 64, 每个sign表示某两行正交与否
	
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int warpsize = size / 4;	// 每行在时间上分出4块，让一个warp的线程依次做累加，再多的话寄存器不够 24
	int warpnum = blockDim.x / warpsize;	// 576/24=24
	int laneid = threadIdx.x % warpsize;	// 0-23
	int warpid = threadIdx.x / warpsize;	// 每个warp中的lanesize个线程协同处理两行的正交化 0-23
	__shared__ int ifConverged; 

	if (tid == 0)
		ifConverged = 1;

	// reset gm_V
	for (int i = tid; i < size*size; i += blockDim.x){
		gm_V[bid * size*size + i] = i/size == i%size;
	}

	__syncthreads();

	double temp_ai[4];	// 将gm_A中的数据取出暂存到寄存器中
	double temp_aj[4];

	__shared__ double my_aiaj[1024];	// threads < 1024
	__shared__ double my_aiai_ajaj[1024];
	__shared__ double sum_aiaj[64];	// k<=64
	
	double vi = 0;
	double vj = 0;
	double tan, cos, sin;
	int indexi = 0;
	int indexj = 0;
	int s = warpsize/2;		// 12
	int iter = 1;
	double tao;
	double signTao;
	int q=0, k=0;
	int test=2;
	while (ifConverged > 0 && test>0)
	{		
		test--;
		iter = 1;
		// 这个循环针对的是roundRobin数组的列，roundRobin的列数即数组A的列的两两组合数
		while (iter < size){		
			// 这个循环针对的是roundRobin数组的行, 循环2次
			for(q=warpid; q<size/2; q+=warpnum)
			{
				my_aiai_ajaj[tid] = 0;
				my_aiaj[tid] = 0;	
				
				indexi = dev_roundRobin[(iter - 1) * size + q];
				indexj = dev_roundRobin[(iter - 1) * size + size - 1 - q];			

				for(k=0; k<4; k++)
				{
					if(laneid + warpsize*k < size)
					{
						temp_ai[k] = dev_G[bid*size*size + indexi*size + laneid+k*warpsize];
						temp_aj[k] = dev_G[bid*size*size + indexj*size + laneid+k*warpsize];
					}
					else
					{
						temp_ai[k] = 0;
						temp_ai[k] = 0;
					}
					my_aiai_ajaj[tid] += temp_ai[k] * temp_ai[k] - temp_aj[k] * temp_aj[k];
					my_aiaj[tid] += temp_ai[k] * temp_aj[k];
				}
				
				__syncthreads();

				// 规约， 规约完毕每个进程都有了同样的 aiai_ajaj 和 aiaj 值（都等于所有进程值的总和）
				if(laneid==0){
					for(k=1; k<warpsize; k++){
						my_aiai_ajaj[tid] += my_aiai_ajaj[tid + k];
						my_aiaj[tid] += my_aiaj[tid + k];
					}

					for(k=1; k<warpsize; k++){
						my_aiai_ajaj[tid + k] = my_aiai_ajaj[tid];
						my_aiaj[tid + k] = my_aiaj[tid];
					}	
				}
				__syncthreads();

				if (my_aiaj[tid] != 0)
				{
					// 计算旋转值（雅可比旋转矩阵）
					tao = my_aiai_ajaj[tid] / (2 * my_aiaj[tid]);
					if (tao > 0)
						signTao = 1;
					if (tao == 0)
						signTao = 1;
					if (tao < 0)
						signTao = -1;
					tan = signTao / ((fabs(tao) + sqrt(1 + tao * tao)));
					cos = 1 / sqrt(1 + tan * tan);
					sin = tan * cos;
					
					// update A
					for(k=0; k<4; k++)
					{
						if (laneid + warpsize*k < size)
						{
							dev_G[bid*size*size + indexi*size + laneid+k*warpsize] =  temp_ai[k] * cos + temp_aj[k] * sin;
							dev_G[bid*size*size + indexj*size + laneid+k*warpsize] = -temp_ai[k] * sin + temp_aj[k] * cos;
						}
					}		
					
					// update V 
					for(k=0; k<4; k++)
					{
						if (laneid + warpsize*k < size)
						{
							vi = gm_V[bid*size*size + (indexi)*size + laneid + k*warpsize] * cos + gm_V[bid*size*size + (indexj)*size + laneid + k*warpsize] * sin;
							vj = -gm_V[bid*size*size + (indexi)*size + laneid + k*warpsize] * sin + gm_V[bid*size*size + (indexj)*size + laneid + k*warpsize] * cos;
							gm_V[bid*size*size + (indexi)*size + laneid + k*warpsize] = vi;
							gm_V[bid*size*size + (indexj)*size + laneid + k*warpsize] = vj;

						}
					}		
				}

				__syncthreads();

				if(laneid==0){
					sum_aiaj[q] = my_aiaj[tid];		// 本行列的正交度
				}

				__syncthreads();
			}		
			iter++;
		}
		
		if (laneid == 0 && warpid < size)
		{
			if (fabs(sum_aiaj[warpid]) < PRECISION)
				sm_sign[warpid] = 0;
			else
				sm_sign[warpid] = 1;

			if (fabs(sum_aiaj[warpid + warpnum]) < PRECISION)
				sm_sign[warpid + warpnum] = 0;
			else
				sm_sign[warpid + warpnum] = 1;
		}

		__syncthreads();	

		if (tid == 0)
		{
			ifConverged = 0;
			for (k = 0; k < size/2; k++)
			{
				ifConverged += sm_sign[k];
			}
		}
		__syncthreads();
	}
}

//dim3 dimGrid9(p, batch, 1);
//dim3 dimBlock9(k, k, 1);
__global__ void myevd_batched_24(double *dev_jointG, int *dev_roundRobin, int p, int k)
{
	__shared__ double shared_G[48][48];
	__shared__ int shared_roundRobin[47][48];
	__shared__ int step;
	__shared__ double shared_V[48][48];
	__shared__ double shared_operators[2][48];
	__shared__ int shared_pairs[2][48];

	//2k*2k
	shared_G[threadIdx.x][threadIdx.y] = dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + threadIdx.y * 2 * k + threadIdx.x];
	shared_G[threadIdx.x + k][threadIdx.y] = dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + threadIdx.y * 2 * k + threadIdx.x + k];
	shared_G[threadIdx.x][threadIdx.y + k] = dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + (threadIdx.y + k) * 2 * k + threadIdx.x];
	shared_G[threadIdx.x + k][threadIdx.y + k] = dev_jointG[blockIdx.y * 2 * k * 2 * k * p + blockIdx.x * 2 * k * 2 * k + (threadIdx.y + k) * 2 * k + threadIdx.x + k];

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

void EVD_(double *dev_jointG, double *dev_A, double *dev_V, int *dev_pairsOfEVD, int p, int q, int height, int width, int *dev_roundRobin, int batch, int k, int slice, int sliceNum, int iter)
{
	if(k>24 && gm_V==NULL){
		cudaMalloc((void **)&gm_V, sizeof(double) * batch * p * 2*k * 2*k);    //<3
	}

	if(k == 1){
		dim3 dimGrid9(p, batch, 1);
		dim3 dimBlock9(2 * k, 2 * k, 1);
		myevd_batched<<<dimGrid9, dimBlock9>>>(dev_jointG, dev_roundRobin, p, k);
	}
	else if (k == 4)
	{
		dim3 dimGrid9(p, batch, 1);
		dim3 dimBlock9(2 * k, 2 * k, 1);
		myevd_batched_4<<<dimGrid9, dimBlock9>>>(dev_jointG, dev_roundRobin, p, k);
	}
	else if (k == 8)
	{
		dim3 dimGrid9(p, batch, 1);
		dim3 dimBlock9(2 * k, 2 * k, 1);
		myevd_batched_8<<<dimGrid9, dimBlock9>>>(dev_jointG, dev_roundRobin, p, k);
	}
	else if (k == 16)
	{	
		dim3 dimGrid9(p, batch, 1);	// 32×100
		dim3 dimBlock9(2 * k, 2 * k, 1);	// 32×32
		myevd_batched_16<<<dimGrid9, dimBlock9>>>(dev_jointG, dev_roundRobin, p, k);
		cudaDeviceSynchronize();
	}
	else if (k == 24)
	{ 
		dim3 dimGrid9(p, batch, 1);	 //
		dim3 dimBlock9(k, k, 1);	// 24×24
		myevd_batched_24<<<dimGrid9, dimBlock9>>>(dev_jointG, dev_roundRobin, p, k);
	}
	else
	{
        // have to use global memory
		dim3 dimGrid9(p, batch, 1);
		dim3 dimBlock9(2 * k, 2 * k, 1);
		mysvd_batched_even_gm_plus<<<p*batch, k/2 * k/2>>>(dev_jointG, 2*k, gm_V, dev_roundRobin);	//height=2*k
	}

	cudaDeviceSynchronize();	

	dim3 dimGrid11(sliceNum, p, batch);
	if(k == 1)
	{
		updateBlockColumn2<<<dimGrid11, 128>>>(dev_A, dev_V, dev_jointG, dev_pairsOfEVD, p, q, height, width, k, slice);
	}
	if (k == 4)
	{
		updateBlockColumn2_4<<<dimGrid11, 128>>>(dev_A, dev_V, dev_jointG, dev_pairsOfEVD, p, q, height, width, k, slice);
	}
	else if (k == 8)
	{
		updateBlockColumn2_8<<<dimGrid11, 128>>>(dev_A, dev_V, dev_jointG, dev_pairsOfEVD, p, q, height, width, k, slice);
	}
	else if (k == 16)
	{
		updateBlockColumn2_16<<<dimGrid11, 256>>>(dev_A, dev_V, dev_jointG, dev_pairsOfEVD, p, q, height, width, k, slice);
	}
	else if (k == 24)
	{
		updateBlockColumn2_24<<<dimGrid11, 256>>>(dev_A, dev_V, dev_jointG, dev_pairsOfEVD, p, q, height, width, k, slice);
	}
	else
	{
        // have to use global memroy
		dim3 dimGrid12(p, batch);
		update_AV<<<dimGrid12, 64>>>(dev_A, dev_V, gm_V, dev_pairsOfEVD, p, height, width, k);
	}
	cudaDeviceSynchronize();

	if(iter==99 && gm_V!=NULL){
		cudaFree(gm_V);
		gm_V = NULL;
		// printf("gm_V freed\n");
	}
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
__global__ void getUDV(double *dev_A, double *dev_U, double *dev_I, double *dev_V, int height, int width, int height0, int width0, int p, int q, double *dev_diag, int minSideLen, int k)
{
	__shared__ double shared_A[32][16];
	__shared__ double sqrtSum[16];
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
		if ((blockIdx.x * k + threadIdx.x) < width0)
		{
			dev_diag[blockIdx.y * minSideLen + blockIdx.x * k + threadIdx.x] = sqrtSum[threadIdx.x];
		}
	}
	__syncthreads();
	if (height0 >= width0)
	{
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
			if ((j * 32 + threadIdx.x) < height0 && (blockIdx.x * k + threadIdx.y) < width0)
			{
				dev_U[blockIdx.y * height0 * height0 + (blockIdx.x * k + threadIdx.y) * height0 + j * 32 + threadIdx.x] = temp2;
			}
			__syncthreads();
		}
		__syncthreads();
		////get V
		for (int j = 0; j < q; j++)
		{
			if ((j * 32 + threadIdx.x) < width0 && (blockIdx.x * k + threadIdx.y) < width0)
			{
				dev_V[blockIdx.y * width0 * width0 + (blockIdx.x * k + threadIdx.y) * width0 + j * 32 + threadIdx.x] = dev_I[blockIdx.y * width * width + (blockIdx.x * k + threadIdx.y) * width + j * 32 + threadIdx.x];
			}
		}
	}
}

//1024*1024
//<<<batch, 1024>>>
__global__ void get_sigma(double *dev_A, double *dev_diag, int height){

/* 	// do fnorm
	__shared__ double all_sum[512];
	int tid = threadIdx.x;
	double sum = 0;
	for(int i=0; i<512; i++){
		sum += dev_A[tid*height + i] * dev_A[tid*height + i];
	}
	all_sum[tid] = sum;

	__syncthreads();

	if(tid==0)
	{
		for(int i=1;i<512;i++)
			all_sum[0] += all_sum[i];
		printf("a fnorm:%lf\n", all_sum[0]);
	} */

	int tid=threadIdx.x;
	int width = blockDim.x;
	// if(tid==0) printf("a101:%lf\n", dev_A[512*9]);
	double sum = 0;
	double temp = 0;
	for(int i=0; i<height; i++){
		temp = dev_A[blockIdx.x * height*width + tid*height + i];
		sum += temp * temp;
	}
	dev_diag[blockIdx.x*width + tid] = sqrt(sum);
}

// <<<1, 1>>>
__global__ void set_d_AG_array(double* dev_A,  double* dev_G, double** d_Aarray, double** d_Barray, double** d_Garray, int batch, int height, int width){
    for(int i=0; i<batch; i++){
        d_Aarray[i] = &dev_A[i*height*width];
        // d_Barray[i] = &dev_A[i*height*width];
        d_Garray[i] = &dev_G[i*width*width];
    }
}
