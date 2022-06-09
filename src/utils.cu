#ifndef UTILS
#define UTILS

#include <iostream>

#define PRECISION 1e-10

/*Print computing device properties*/
void printDeviceProp(const cudaDeviceProp &prop)    //&2.2
{
	printf("Device Name : %s.\n", prop.name);
	printf("totalGlobalMem : %ld.\n", prop.totalGlobalMem);
	printf("sharedMemPerBlock : %ld.\n", prop.sharedMemPerBlock);
	printf("regsPerBlock : %d.\n", prop.regsPerBlock);
	printf("warpSize : %d.\n", prop.warpSize);
	printf("memPitch : %ld.\n", prop.memPitch);
	printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
	printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("totalConstMem : %ld.\n", prop.totalConstMem);
	printf("major.minor : %d.%d.\n", prop.major, prop.minor);
	printf("clockRate : %d.\n", prop.clockRate);
	printf("textureAlignment : %ld.\n", prop.textureAlignment);
	printf("deviceOverlap : %d.\n", prop.deviceOverlap);
	printf("multiProcessorCount : %d.\n\n", prop.multiProcessorCount);

	printf("maxThreadsPerMultiProcessor : %d\n", prop.maxThreadsPerMultiProcessor);
	printf("sharedMemPerMultiprocessor : %d\n", prop.sharedMemPerMultiprocessor);
	printf("regsPerMultiprocessor : %d\n", prop.regsPerMultiprocessor);
}

/*Obtain computing device information and initialize the computing device*/
bool initCUDA()
{
	int count;
	cudaGetDeviceCount(&count);
	if (count == 0)
	{
		printf("There is no device.\n");
		return false;
	}
    else
    {
        printf("Find the device successfully.\n");
    }
	int i;
	for (i = 0; i < count; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		// printf("index:%d\n", i);
		printDeviceProp(prop);
	}
	//set its value between 0 and 7 if there are 8 v100
	cudaSetDevice(0);   
	return true;
}

void setCUDAConfig()
{
	if(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) != cudaSuccess)  //&2.2
        printf("Error");
    else
        printf("Bank size has been set successfully.\n");
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

// dimBlock(32, 32, 1);
__global__ void generate_roundRobin_128(int *dev_roundRobin, int n)
{
	// if(threadIdx.x==0&&threadIdx.y==0)
	// 	printf("hello!!!!\n");
	int i = threadIdx.x;
	int j = threadIdx.y;
	int tid = i*32 + j;
	__shared__ int firstline[1024];

	if(tid < n)
		firstline[tid] = tid*2 - (tid<n/2 ? 0 : ((tid - n/2) * 4 + 1));
	
	while(i<n-1)
	{
		while(j<n)
		{
			int flag = (j - i + n) % n;
			int picked = flag;
			if(flag <= n/2)
			{
				if(j > n/2 || flag == n/2)
					picked = (flag - 1 + n) % n;
				else if(j < flag)
					picked = (flag - 1 + n) % n;
			}
			else if(j>n/2 && j<flag)
				picked = (flag - 1 + n) % n;
			if(j==n/2)
				picked = j;
			dev_roundRobin[i*n+j] = firstline[picked];
			j += 32;
		}
		j = threadIdx.y;
		i += 32;
	}
}

int orth_matrix_verify(double* m, int size, int tag=0){
    double sum = 0;
    for(int i=0; i<size; i++){
        for(int j=0; j<size; j++){
            sum = 0;
            for(int k=0; k<size; k++){
                sum += m[i*size + k] * m[j*size + k];
            }
			if(i!=j && (sum-0>PRECISION || 0-sum>PRECISION)){
				printf("matrix %d is not orthogonal\n", tag);
				return 0;
			}
        }
    }
	printf("matrix %d is orthogonal\n", tag);
	return 1;
}

// <<<1, 1024>>> // less than 1024
__global__ void transform(double *dev_A0, double* dev_A1, int height, int width){
	int tid = threadIdx.x;
	int bid = blockIdx.x;	// batch index
	for(int i=0; i<height; i++){
		if(tid < width)
			dev_A1[bid*height*width + i*width + tid] = dev_A0[bid*height*width + tid*height + i];
	}
}

#endif