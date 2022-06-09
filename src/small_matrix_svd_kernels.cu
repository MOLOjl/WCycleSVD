#include <string>
#include<cuda.h>
#include<cuda_runtime.h>
#define PRECISION 1e-10
using namespace std;

// <<<1,32>>>
// <<<batch, 16 * width0/2 >> >
__global__ void small_svd_even_column(double *dev_A, int height0, int width0, double *dev_U, double *dev_V, double *dev_diag, int *dev_roundRobin)
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
	
	// init V
	if (tid < 16)
	{	
		sm_V[laneid][laneid] = 1;
		sm_V[laneid + 16][laneid + 16] = 1;
		sm_V[laneid + 32][laneid + 32] = 1;
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
	int test = 10;
	while (ifConverged > 0 && test>0)
	{
		iter = 1;
		test--;
		while (iter < width0)
		{
			indexi = sm_roundRobin[(iter - 1) * width0 + warpid];	// [0-0] 0, [1-1] 2
			indexj = sm_roundRobin[(iter - 1) * width0 + width0 - 1 - warpid]; // [3-3] 1, ]2-2] 3
			ai0 = sm_A[indexi][laneid];		//[0-0][0-15], [2-2][0-15]
			aj0 = sm_A[indexj][laneid];		// [1-1][0-15], [3-3][0-15]
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
					signTao = 1;
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
		
		if (laneid == 0)
		{
			if (fabs(aiaj) <= PRECISION * fabs(aiai_ajaj))
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

	double aii0 = 0, aii1 = 0, aii2 = 0;
	if (laneid < height0)
		aii0 = sm_A[warpid][laneid];
	if ((laneid + 16) < height0)
		aii1 = sm_A[warpid][laneid + 16];
	if ((laneid + 32) < height0)
		aii2 = sm_A[warpid][laneid + 32];
	double form = aii0 * aii0 + aii1 * aii1 + aii2 * aii2;
	for (s = 8; s > 0; s /= 2)
	{
		form += __shfl_xor_sync(-1, form, s);
	}
	form = sqrt(form);
	if (laneid < height0)
		dev_U[bid * height0 * height0 + warpid * height0 + laneid] = aii0 / form;
	if ((laneid + 16) < height0)
		dev_U[bid * height0 * height0 + warpid * height0 + laneid + 16] = aii1 / form;
	if ((laneid + 32) < height0)
		dev_U[bid * height0 * height0 + warpid * height0 + laneid + 32] = aii2 / form;
	if (laneid == 0)
		dev_diag[bid * width0 + warpid] = form;

	if (laneid < height0)
		aii0 = sm_A[warpid + width0 / 2][laneid];
	if ((laneid + 16) < height0)
		aii1 = sm_A[warpid + width0 / 2][laneid + 16];
	if ((laneid + 32) < height0)
		aii2 = sm_A[warpid + width0 / 2][laneid + 32];
	form = aii0 * aii0 + aii1 * aii1 + aii2 * aii2;
	for (s = 8; s > 0; s /= 2)
	{
		form += __shfl_xor_sync(-1, form, s);
	}
	form = sqrt(form);

	if (laneid < height0)
		dev_U[bid * height0 * height0 + (warpid + width0 / 2) * height0 + laneid] = aii0 / form;
	if ((laneid + 16) < height0)
		dev_U[bid * height0 * height0 + (warpid + width0 / 2) * height0 + laneid + 16] = aii1 / form;
	if ((laneid + 32) < height0)
		dev_U[bid * height0 * height0 + (warpid + width0 / 2) * height0 + laneid + 32] = aii2 / form;
	if (laneid == 0)
		dev_diag[bid * width0 + width0 / 2 + warpid] = form;

	__syncthreads();

	if (laneid < width0)
		dev_V[bid * width0 * width0 + warpid * width0 + laneid] = sm_V[warpid][laneid];
	if ((laneid + 16) < width0)
		dev_V[bid * width0 * width0 + warpid * width0 + laneid + 16] = sm_V[warpid][laneid + 16];
	if ((laneid + 32) < width0)
		dev_V[bid * width0 * width0 + warpid * width0 + laneid + 32] = sm_V[warpid][laneid + 32];
	__syncthreads();
	if (laneid < width0)
		dev_V[bid * width0 * width0 + (warpid + width0 / 2) * width0 + laneid] = sm_V[warpid + width0 / 2][laneid];
	if ((laneid + 16) < width0)
		dev_V[bid * width0 * width0 + (warpid + width0 / 2) * width0 + laneid + 16] = sm_V[warpid + width0 / 2][laneid + 16];
	if ((laneid + 32) < width0)
		dev_V[bid * width0 * width0 + (warpid + width0 / 2) * width0 + laneid + 32] = sm_V[warpid + width0 / 2][laneid + 32];
	__syncthreads();

}

// <<<batch, 16 * (width0 + 1) / 2 >>>
__global__ void small_svd_odd_column(double *dev_A, int height0, int width0, double *dev_U, double *dev_V, double *dev_diag, int *dev_roundRobin)
{
	__shared__ double sm_A[48][48];
	__shared__ double sm_V[48][48];
	__shared__ int sm_roundRobin[47 * 48];
	__shared__ int sm_sign[24];
	__shared__ int ifConverged;
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int laneid = threadIdx.x % 16;
	int warpid = threadIdx.x / 16;

	for (int i = tid; i < width0 * height0; i += blockDim.x)
	{
		sm_A[i / height0][i % height0] = dev_A[bid * height0 * width0 + i];
	}
	if (tid == 0)
		ifConverged = 1;

	if (tid < 16)
	{
		sm_V[laneid][laneid] = 1;
		sm_V[laneid + 16][laneid + 16] = 1;
		sm_V[laneid + 32][laneid + 32] = 1;
	}

	for (int i = tid; i < width0 * (width0 + 1); i += blockDim.x)
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
	while (ifConverged > 0)
	{
		iter = 1;
		while (iter < (width0 + 1))
		{
			if (warpid < (width0 - 1) / 2)
			{
				indexi = sm_roundRobin[(iter - 1) * (width0 + 1) + warpid];
				indexj = sm_roundRobin[(iter - 1) * (width0 + 1) + width0 - warpid];

				ai0 = sm_A[indexi][laneid];
				aj0 = sm_A[indexj][laneid];
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
			}
			if (warpid < (((width0 - 1) / 2 - 1) / 2 + 1) * 2)
			{
				for (s = 8; s > 0; s /= 2)
				{
					aiai_ajaj += __shfl_xor_sync(0xffffffff, aiai_ajaj, s);
					aiaj += __shfl_xor_sync(0xffffffff, aiaj, s);
				}
			}
			if (warpid < ((width0 - 1) / 2))
			{
				if (aiaj != 0)
				{
					tao = aiai_ajaj / (2 * aiaj);
					if (tao > 0)
						signTao = 1;
					if (tao == 0)
						signTao = 1;
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
			}
			__syncthreads();
			iter++;
		}
		if (warpid < width0 / 2 && laneid == 0)
		{
			if (fabs(aiaj) <= PRECISION * fabs(aiai_ajaj))
				sm_sign[warpid] = 0;
			else
				sm_sign[warpid] = 1;
		}
		__syncthreads();
		if (warpid == 0 && laneid == 0)
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

	/////////////////////////////new
	double aii0 = 0, aii1 = 0, aii2 = 0;
	if (laneid < height0)
		aii0 = sm_A[warpid][laneid];
	if ((laneid + 16) < height0)
		aii1 = sm_A[warpid][laneid + 16];
	if ((laneid + 32) < height0)
		aii2 = sm_A[warpid][laneid + 32];
	double form = aii0 * aii0 + aii1 * aii1 + aii2 * aii2;
	for (s = 8; s > 0; s /= 2)
	{
		form += __shfl_xor_sync(-1, form, s);
	}
	form = sqrt(form);
	if (laneid < height0)
		dev_U[bid * height0 * height0 + warpid * height0 + laneid] = aii0 / form;
	if ((laneid + 16) < height0)
		dev_U[bid * height0 * height0 + warpid * height0 + laneid + 16] = aii1 / form;
	if ((laneid + 32) < height0)
		dev_U[bid * height0 * height0 + warpid * height0 + laneid + 32] = aii2 / form;
	if (laneid == 0)
		dev_diag[bid * width0 + warpid] = form;

	if (warpid < ((width0 - 1) / 2))
	{
		if (laneid < height0)
			aii0 = sm_A[warpid + (width0 + 1) / 2][laneid];
		if ((laneid + 16) < height0)
			aii1 = sm_A[warpid + (width0 + 1) / 2][laneid + 16];
		if ((laneid + 32) < height0)
			aii2 = sm_A[warpid + (width0 + 1) / 2][laneid + 32];
		form = aii0 * aii0 + aii1 * aii1 + aii2 * aii2;
		for (s = 8; s > 0; s /= 2)
		{
			form += __shfl_xor_sync(-1, form, s);
		}
		form = sqrt(form);
		if (laneid < height0)
			dev_U[bid * height0 * height0 + (warpid + (width0 + 1) / 2) * height0 + laneid] = aii0 / form;
		if ((laneid + 16) < height0)
			dev_U[bid * height0 * height0 + (warpid + (width0 + 1) / 2) * height0 + laneid + 16] = aii1 / form;
		if ((laneid + 32) < height0)
			dev_U[bid * height0 * height0 + (warpid + (width0 + 1) / 2) * height0 + laneid + 32] = aii2 / form;
		if (laneid == 0)
			dev_diag[bid * width0 + (width0 + 1) / 2 + warpid] = form;
	}
	__syncthreads();

	if (laneid < width0)
		dev_V[bid * width0 * width0 + warpid * width0 + laneid] = sm_V[warpid][laneid];
	if ((laneid + 16) < width0)
		dev_V[bid * width0 * width0 + warpid * width0 + laneid + 16] = sm_V[warpid][laneid + 16];
	if ((laneid + 32) < width0)
		dev_V[bid * width0 * width0 + warpid * width0 + laneid + 32] = sm_V[warpid][laneid + 32];
	__syncthreads();
	if (warpid < ((width0 - 1) / 2))
	{
		if (laneid < width0)
			dev_V[bid * width0 * width0 + (warpid + (width0 + 1) / 2) * width0 + laneid] = sm_V[warpid + (width0 + 1) / 2][laneid];
		if ((laneid + 16) < width0)
			dev_V[bid * width0 * width0 + (warpid + (width0 + 1) / 2) * width0 + laneid + 16] = sm_V[warpid + (width0 + 1) / 2][laneid + 16];
		if ((laneid + 32) < width0)
			dev_V[bid * width0 * width0 + (warpid + (width0 + 1) / 2) * width0 + laneid + 32] = sm_V[warpid + (width0 + 1) / 2][laneid + 32];
	}
	__syncthreads();
}

// <<<batch, 16 * height0/2 >>>
__global__ void small_svd_even_column_trans(double *dev_A, int height0, int width0, double *dev_U, double *dev_V, double *dev_diag, int *dev_roundRobin)
{
	__shared__ double sm_A[48][48];
	__shared__ double sm_V[48][48];
	__shared__ int sm_roundRobin[47 * 48];
	__shared__ int sm_sign[24];
	__shared__ int ifConverged;
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int laneid = threadIdx.x % 16;
	int warpid = threadIdx.x / 16;

	for (int i = tid; i < height0 * width0; i += blockDim.x)
	{
		sm_A[i % height0][i / height0] = dev_A[bid * height0 * width0 + i];
	}
	if (tid == 0)
		ifConverged = 1;

	if (tid < 16)
	{
		sm_V[laneid][laneid] = 1;
		sm_V[laneid + 16][laneid + 16] = 1;
		sm_V[laneid + 32][laneid + 32] = 1;
	}

	for (int i = tid; i < (height0 - 1) * height0; i += blockDim.x)
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
	while (ifConverged > 0)
	{
		iter = 1;
		while (iter < height0)
		{
			indexi = sm_roundRobin[(iter - 1) * height0 + warpid];
			indexj = sm_roundRobin[(iter - 1) * height0 + height0 - 1 - warpid];
			ai0 = sm_A[indexi][laneid];
			aj0 = sm_A[indexj][laneid];
			if ((laneid + 16) < width0)
			{
				ai1 = sm_A[indexi][laneid + 16];
				aj1 = sm_A[indexj][laneid + 16];
			}
			else
			{
				ai1 = 0;
				aj1 = 0;
			}
			if ((laneid + 32) < width0)
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

			if (warpid < ((height0 / 2 - 1) / 2 + 1) * 2)
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
					signTao = 1;
				if (tao < 0)
					signTao = -1;
				tan = signTao / ((fabs(tao) + sqrt(1 + tao * tao)));
				cos = 1 / sqrt(1 + tan * tan);
				sin = tan * cos;
				//update A
				sm_A[indexi][laneid] = ai0 * cos + aj0 * sin;
				sm_A[indexj][laneid] = -ai0 * sin + aj0 * cos;
				if ((laneid + 16) < width0)
				{
					sm_A[indexi][laneid + 16] = ai1 * cos + aj1 * sin;
					sm_A[indexj][laneid + 16] = -ai1 * sin + aj1 * cos;
				}
				if ((laneid + 32) < width0)
				{
					sm_A[indexi][laneid + 32] = ai2 * cos + aj2 * sin;
					sm_A[indexj][laneid + 32] = -ai2 * sin + aj2 * cos;
				}

				//update V ///////////////////////////////////////////////////////////////////////////
				if (laneid < height0)
				{
					vi = sm_V[indexi][laneid] * cos + sm_V[indexj][laneid] * sin;
					vj = -sm_V[indexi][laneid] * sin + sm_V[indexj][laneid] * cos;
					sm_V[indexi][laneid] = vi;
					sm_V[indexj][laneid] = vj;
				}
				if ((laneid + 16) < height0)
				{
					vi = sm_V[indexi][laneid + 16] * cos + sm_V[indexj][laneid + 16] * sin;
					vj = -sm_V[indexi][laneid + 16] * sin + sm_V[indexj][laneid + 16] * cos;
					sm_V[indexi][laneid + 16] = vi;
					sm_V[indexj][laneid + 16] = vj;
				}
				if ((laneid + 32) < height0)
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
			if (fabs(aiaj) <= PRECISION * fabs(aiai_ajaj))
				sm_sign[warpid] = 0;
			else
				sm_sign[warpid] = 1;
		}
		__syncthreads();
		if (tid == 0)
		{
			ifConverged = 0;
			for (int i = 0; i < height0 / 2; i++)
			{
				ifConverged += sm_sign[i];
			}
		}
		__syncthreads();

	}
	__syncthreads();
	double aii0 = 0, aii1 = 0, aii2 = 0;
	if (laneid < width0)
		aii0 = sm_A[warpid][laneid];
	if ((laneid + 16) < width0)
		aii1 = sm_A[warpid][laneid + 16];
	if ((laneid + 32) < width0)
		aii2 = sm_A[warpid][laneid + 32];
	double form = aii0 * aii0 + aii1 * aii1 + aii2 * aii2;
	for (s = 8; s > 0; s /= 2)
	{
		form += __shfl_xor_sync(-1, form, s);
	}
	form = sqrt(form);
	if (laneid < width0)
		dev_V[bid * width0 * width0 + warpid * width0 + laneid] = aii0 / form;
	if ((laneid + 16) < width0)
		dev_V[bid * width0 * width0 + warpid * width0 + (laneid + 16)] = aii1 / form;
	if ((laneid + 32) < width0)
		dev_V[bid * width0 * width0 + warpid * width0 + (laneid + 32)] = aii2 / form;
	if (laneid == 0)
		dev_diag[bid * height0 + warpid] = form;

	if (laneid < width0)
		aii0 = sm_A[warpid + height0 / 2][laneid];
	if ((laneid + 16) < width0)
		aii1 = sm_A[warpid + height0 / 2][laneid + 16];
	if ((laneid + 32) < width0)
		aii2 = sm_A[warpid + height0 / 2][laneid + 32];
	form = aii0 * aii0 + aii1 * aii1 + aii2 * aii2;
	for (s = 8; s > 0; s /= 2)
	{
		form += __shfl_xor_sync(-1, form, s);
	}
	form = sqrt(form);
	if (laneid < width0)
		dev_V[bid * width0 * width0 + (warpid + height0 / 2) * width0 + laneid] = aii0 / form;
	if ((laneid + 16) < width0)
		dev_V[bid * width0 * width0 + (warpid + height0 / 2) * width0 + laneid + 16] = aii1 / form;
	if ((laneid + 32) < width0)
		dev_V[bid * width0 * width0 + (warpid + height0 / 2) * width0 + laneid + 32] = aii2 / form;
	if (laneid == 0)
		dev_diag[bid * height0 + height0 / 2 + warpid] = form;

	__syncthreads();

	if (laneid < height0)
		dev_U[bid * height0 * height0 + warpid * height0 + laneid] = sm_V[warpid][laneid];
	
	
	if ((laneid + 16) < height0)
		dev_U[bid * height0 * height0 + warpid * height0 + (laneid + 16)] = sm_V[warpid][laneid + 16];
	if ((laneid + 32) < height0)
		dev_U[bid * height0 * height0 + warpid * height0 + (laneid + 32)] = sm_V[warpid][laneid + 32];
	__syncthreads();
	if (laneid < height0)
		dev_U[bid * height0 * height0 + (warpid + height0 / 2) * height0 + laneid] = sm_V[warpid + height0 / 2][laneid];
	if ((laneid + 16) < height0)
		dev_U[bid * height0 * height0 + (warpid + height0 / 2) * height0 + laneid + 16] = sm_V[warpid + height0 / 2][laneid + 16];
	if ((laneid + 32) < height0)
		dev_U[bid * height0 * height0 + (warpid + height0 / 2) * height0 + laneid + 32] = sm_V[warpid + height0 / 2][laneid + 32];
	__syncthreads();
}

// <<<batch, 16 * (width0 + 1) / 2 >>>
__global__ void small_svd_odd_column_trans(double *dev_A, int height0, int width0, double *dev_U, double *dev_V, double *dev_diag, int *dev_roundRobin)
{
	__shared__ double sm_A[48][48];
	__shared__ double sm_V[48][48];
	__shared__ int sm_roundRobin[47 * 48];
	__shared__ int sm_sign[24];
	__shared__ int ifConverged;
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int laneid = threadIdx.x % 16;
	int warpid = threadIdx.x / 16;

	for (int i = tid; i < height0 * width0; i += blockDim.x)
	{
		sm_A[i % height0][i / height0] = dev_A[bid * height0 * width0 + i];
	}
	if (tid == 0)
		ifConverged = 1;

	if (tid < 16)
	{
		sm_V[laneid][laneid] = 1;
		sm_V[laneid + 16][laneid + 16] = 1;
		sm_V[laneid + 32][laneid + 32] = 1;
	}

	for (int i = tid; i < height0 * (height0 + 1); i += blockDim.x)
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
	while (ifConverged > 0)
	{
		iter = 1;
		while (iter < (height0 + 1))
		{
			if (warpid < (height0 - 1) / 2)
			{
				indexi = sm_roundRobin[(iter - 1) * (height0 + 1) + warpid];
				indexj = sm_roundRobin[(iter - 1) * (height0 + 1) + height0 - warpid];

				ai0 = sm_A[indexi][laneid];
				aj0 = sm_A[indexj][laneid];
				if ((laneid + 16) < width0)
				{
					ai1 = sm_A[indexi][laneid + 16];
					aj1 = sm_A[indexj][laneid + 16];
				}
				else
				{
					ai1 = 0;
					aj1 = 0;
				}
				if ((laneid + 32) < width0)
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
			}
			if (warpid < (((height0 - 1) / 2 - 1) / 2 + 1) * 2)
			{
				for (s = 8; s > 0; s /= 2)
				{
					aiai_ajaj += __shfl_xor_sync(0xffffffff, aiai_ajaj, s);
					aiaj += __shfl_xor_sync(0xffffffff, aiaj, s);
				}
			}
			if (warpid < ((height0 - 1) / 2))
			{
				if (aiaj != 0)
				{
					tao = aiai_ajaj / (2 * aiaj);
					if (tao > 0)
						signTao = 1;
					if (tao == 0)
						signTao = 1;
					if (tao < 0)
						signTao = -1;
					tan = signTao / ((fabs(tao) + sqrt(1 + tao * tao)));
					cos = 1 / sqrt(1 + tan * tan);
					sin = tan * cos;
					//update A
					sm_A[indexi][laneid] = ai0 * cos + aj0 * sin;
					sm_A[indexj][laneid] = -ai0 * sin + aj0 * cos;
					if ((laneid + 16) < width0)
					{
						sm_A[indexi][laneid + 16] = ai1 * cos + aj1 * sin;
						sm_A[indexj][laneid + 16] = -ai1 * sin + aj1 * cos;
					}
					if ((laneid + 32) < width0)
					{
						sm_A[indexi][laneid + 32] = ai2 * cos + aj2 * sin;
						sm_A[indexj][laneid + 32] = -ai2 * sin + aj2 * cos;
					}

					//update V ///////////////////////////////////////////////////////////////////////////
					if (laneid < height0)
					{
						vi = sm_V[indexi][laneid] * cos + sm_V[indexj][laneid] * sin;
						vj = -sm_V[indexi][laneid] * sin + sm_V[indexj][laneid] * cos;
						sm_V[indexi][laneid] = vi;
						sm_V[indexj][laneid] = vj;
					}
					if ((laneid + 16) < height0)
					{
						vi = sm_V[indexi][laneid + 16] * cos + sm_V[indexj][laneid + 16] * sin;
						vj = -sm_V[indexi][laneid + 16] * sin + sm_V[indexj][laneid + 16] * cos;
						sm_V[indexi][laneid + 16] = vi;
						sm_V[indexj][laneid + 16] = vj;
					}
					if ((laneid + 32) < height0)
					{
						vi = sm_V[indexi][laneid + 32] * cos + sm_V[indexj][laneid + 32] * sin;
						vj = -sm_V[indexi][laneid + 32] * sin + sm_V[indexj][laneid + 32] * cos;
						sm_V[indexi][laneid + 32] = vi;
						sm_V[indexj][laneid + 32] = vj;
					}
				}
			}
			__syncthreads();
			iter++;
		}
		if (warpid < height0 / 2 && laneid == 0)
		{
			if (fabs(aiaj) <= PRECISION * fabs(aiai_ajaj))
				sm_sign[warpid] = 0;
			else
				sm_sign[warpid] = 1;
		}
		__syncthreads();
		if (warpid == 0 && laneid == 0)
		{
			ifConverged = 0;
			for (int i = 0; i < height0 / 2; i++)
			{
				ifConverged += sm_sign[i];
			}
		}
		__syncthreads();
	}
	__syncthreads();

	/////////////////////////////new
	double aii0 = 0, aii1 = 0, aii2 = 0;
	if (laneid < width0)
		aii0 = sm_A[warpid][laneid];
	if ((laneid + 16) < width0)
		aii1 = sm_A[warpid][laneid + 16];
	if ((laneid + 32) < width0)
		aii2 = sm_A[warpid][laneid + 32];
	double form = aii0 * aii0 + aii1 * aii1 + aii2 * aii2;
	for (s = 8; s > 0; s /= 2)
	{
		form += __shfl_xor_sync(-1, form, s);
	}
	form = sqrt(form);
	if (laneid < width0)
		dev_V[bid * width0 * width0 + warpid * width0 + laneid] = aii0 / form;
	if ((laneid + 16) < width0)
		dev_V[bid * width0 * width0 + warpid * width0 + laneid + 16] = aii1 / form;
	if ((laneid + 32) < width0)
		dev_V[bid * width0 * width0 + warpid * width0 + laneid + 32] = aii2 / form;
	if (laneid == 0)
		dev_diag[bid * height0 + warpid] = form;

	if (warpid < ((height0 - 1) / 2))
	{
		if (laneid < width0)
			aii0 = sm_A[warpid + (height0 + 1) / 2][laneid];
		if ((laneid + 16) < width0)
			aii1 = sm_A[warpid + (height0 + 1) / 2][laneid + 16];
		if ((laneid + 32) < width0)
			aii2 = sm_A[warpid + (height0 + 1) / 2][laneid + 32];
		form = aii0 * aii0 + aii1 * aii1 + aii2 * aii2;
		for (s = 8; s > 0; s /= 2)
		{
			form += __shfl_xor_sync(-1, form, s);
		}
		form = sqrt(form);
		if (laneid < width0)
			dev_V[bid * width0 * width0 + (warpid + (height0 + 1) / 2) * width0 + laneid] = aii0 / form;
		if ((laneid + 16) < width0)
			dev_V[bid * width0 * width0 + (warpid + (height0 + 1) / 2) * width0 + laneid + 16] = aii1 / form;
		if ((laneid + 32) < width0)
			dev_V[bid * width0 * width0 + (warpid + (height0 + 1) / 2) * width0 + laneid + 32] = aii2 / form;
		if (laneid == 0)
			dev_diag[bid * height0 + (height0 + 1) / 2 + warpid] = form;
	}
	__syncthreads();

	if (laneid < height0)
		dev_U[bid * height0 * height0 + warpid * height0 + laneid] = sm_V[warpid][laneid];
	if ((laneid + 16) < height0)
		dev_U[bid * height0 * height0 + warpid * height0 + (laneid + 16)] = sm_V[warpid][laneid + 16];
	if ((laneid + 32) < height0)
		dev_U[bid * height0 * height0 + warpid * height0 + (laneid + 32)] = sm_V[warpid][laneid + 32];
	__syncthreads();
	if (warpid < ((height0 - 1) / 2))
	{
		if (laneid < height0)
			dev_U[bid * height0 * height0 + (warpid + (height0 + 1) / 2) * height0 + laneid] = sm_V[warpid + (height0 + 1) / 2][laneid];
		if ((laneid + 16) < height0)
			dev_U[bid * height0 * height0 + (warpid + (height0 + 1) / 2) * height0 + laneid + 16] = sm_V[warpid + (height0 + 1) / 2][laneid + 16];
		if ((laneid + 32) < height0)
			dev_U[bid * height0 * height0 + (warpid + (height0 + 1) / 2) * height0 + laneid + 32] = sm_V[warpid + (height0 + 1) / 2][laneid + 32];
	}
	__syncthreads();
}

// <<<1,32>>>
// <<<batch, 16>>>
__global__ void small_svd_even_column_1_warp(double *dev_A, int height0, int width0, double *dev_U, double *dev_V, double *dev_diag, int *dev_roundRobin)
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
	
	// init V
	if (tid < 16)
	{	
		sm_V[laneid][laneid] = 1;
		sm_V[laneid + 16][laneid + 16] = 1;
		sm_V[laneid + 32][laneid + 32] = 1;
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
	int test = 10;
	while (ifConverged > 0 && test>0)
	{
		iter = 1;
		test--;
		while (iter < width0)
		{
			warpid = threadIdx.x / 16; // equivalent to 0
			while(warpid < width0/2){
				
				indexi = sm_roundRobin[(iter - 1) * width0 + warpid];	// [0-0] 0, [1-1] 2
				indexj = sm_roundRobin[(iter - 1) * width0 + width0 - 1 - warpid]; // [3-3] 1, ]2-2] 3
				ai0 = sm_A[indexi][laneid];		//[0-0][0-15], [2-2][0-15]
				aj0 = sm_A[indexj][laneid];		// [1-1][0-15], [3-3][0-15]
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
						signTao = 1;
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

				if(iter == width0-1){
					if (laneid == 0)
					{
						if (fabs(aiaj) <= PRECISION * fabs(aiai_ajaj))
							sm_sign[warpid] = 0;
						else
							sm_sign[warpid] = 1;
					}			
				}
				__syncthreads();
				warpid ++;
			}
			
			iter++;			
		}

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

	// update U Sig V
	warpid = threadIdx.x / 16; // equivalent to 0
	while(warpid < width0/2){
		double aii0 = 0, aii1 = 0, aii2 = 0;
		if (laneid < height0)
			aii0 = sm_A[warpid][laneid];
		if ((laneid + 16) < height0)
			aii1 = sm_A[warpid][laneid + 16];
		if ((laneid + 32) < height0)
			aii2 = sm_A[warpid][laneid + 32];
		double form = aii0 * aii0 + aii1 * aii1 + aii2 * aii2;
		for (s = 8; s > 0; s /= 2)
		{
			form += __shfl_xor_sync(-1, form, s);
		}
		form = sqrt(form);
		if (laneid < height0)
			dev_U[bid * height0 * height0 + warpid * height0 + laneid] = aii0 / form;
		if ((laneid + 16) < height0)
			dev_U[bid * height0 * height0 + warpid * height0 + laneid + 16] = aii1 / form;
		if ((laneid + 32) < height0)
			dev_U[bid * height0 * height0 + warpid * height0 + laneid + 32] = aii2 / form;
		if (laneid == 0)
			dev_diag[bid * width0 + warpid] = form;

		if (laneid < height0)
			aii0 = sm_A[warpid + width0 / 2][laneid];
		if ((laneid + 16) < height0)
			aii1 = sm_A[warpid + width0 / 2][laneid + 16];
		if ((laneid + 32) < height0)
			aii2 = sm_A[warpid + width0 / 2][laneid + 32];
		form = aii0 * aii0 + aii1 * aii1 + aii2 * aii2;
		for (s = 8; s > 0; s /= 2)
		{
			form += __shfl_xor_sync(-1, form, s);
		}
		form = sqrt(form);

		if (laneid < height0)
			dev_U[bid * height0 * height0 + (warpid + width0 / 2) * height0 + laneid] = aii0 / form;
		if ((laneid + 16) < height0)
			dev_U[bid * height0 * height0 + (warpid + width0 / 2) * height0 + laneid + 16] = aii1 / form;
		if ((laneid + 32) < height0)
			dev_U[bid * height0 * height0 + (warpid + width0 / 2) * height0 + laneid + 32] = aii2 / form;
		if (laneid == 0)
			dev_diag[bid * width0 + width0 / 2 + warpid] = form;

		__syncthreads();

		if (laneid < width0)
			dev_V[bid * width0 * width0 + warpid * width0 + laneid] = sm_V[warpid][laneid];
		if ((laneid + 16) < width0)
			dev_V[bid * width0 * width0 + warpid * width0 + laneid + 16] = sm_V[warpid][laneid + 16];
		if ((laneid + 32) < width0)
			dev_V[bid * width0 * width0 + warpid * width0 + laneid + 32] = sm_V[warpid][laneid + 32];
		__syncthreads();
		if (laneid < width0)
			dev_V[bid * width0 * width0 + (warpid + width0 / 2) * width0 + laneid] = sm_V[warpid + width0 / 2][laneid];
		if ((laneid + 16) < width0)
			dev_V[bid * width0 * width0 + (warpid + width0 / 2) * width0 + laneid + 16] = sm_V[warpid + width0 / 2][laneid + 16];
		if ((laneid + 32) < width0)
			dev_V[bid * width0 * width0 + (warpid + width0 / 2) * width0 + laneid + 32] = sm_V[warpid + width0 / 2][laneid + 32];
		
		__syncthreads();
		warpid ++;	
	}
}
