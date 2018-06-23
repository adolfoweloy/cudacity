#include "reduce.h"

/***************************************************************************
description: device function to retrieve absolute value
source.....: http://graphics.stanford.edu/~seander/bithacks.html#IntegerAbs
***************************************************************************/
__device__ int dabs(size_t a)
{
    int const mask = a >> sizeof(int) * CHAR_BIT - 1;
    return (a + mask) ^ mask;
}

/***************************************************************************
description: device function to retrieve min value given two values
***************************************************************************/
__device__ int dmin(size_t a, size_t b)
{
    return (dabs(a+b) - dabs(a-b))/2;
}

/***************************************************************************
description: reduce function by min values using interleaved memory address
***************************************************************************/
__global__ void dreduce(size_t *input, size_t *output, size_t pitch)
{
    // using dynamic shared memory
    extern __shared__ size_t sdata[];

    // thread id that will be used to identify the matrix index
    size_t tid = threadIdx.x;

    // copying all data to shared memory
    size_t *size_t_data = sdata;
    for (int k = 0; k < M; ++k) {
        size_t *row = (size_t*) &size_t_data[tid * M];
        row[k] = input[tid * (pitch / sizeof(size_t)) + k];
    }

    __syncthreads();

    // reducing using interleaved addressing
    for (size_t s = 1; s < blockDim.x; s *= 2)
    {
        size_t *r1 = (size_t*) &size_t_data[tid * M];
        size_t *r2 = (size_t*) &size_t_data[(tid + s) * M];
        if (tid % (2 * s) == 0)
        {
            for (int idx = 0; idx < M; ++idx)
                r1[idx] = dmin(r1[idx], r2[idx]);
        }
        __syncthreads();
    }

    // writing the result
    if (tid == 0) {
        size_t *r = (size_t*) &size_t_data[tid];
        for (int idx = 0; idx < M; ++idx)
            output[idx] = r[idx];
    }
}

/***************************************************************************
description: check for cuda errors
***************************************************************************/
static void _gpu_check(cudaError error)
{
    if (error != cudaSuccess)
    {
        printf("GPU error %d: %s\n", error, cudaGetErrorString(error));
        exit(-1);
    }
}

/***************************************************************************
description: public interface for parallel reduction
***************************************************************************/
void reduce(size_t list_size, size_t **h_list, size_t *h_output)
{
    // device defs
    size_t *d_list;
    size_t *d_output;

    size_t pitch;
    _gpu_check(cudaMallocPitch(&d_list, &pitch, N * N * sizeof(size_t), list_size));
    _gpu_check(cudaMemcpy2D(d_list, pitch, h_list, N * N * sizeof(size_t), N * N * sizeof(size_t), list_size, cudaMemcpyHostToDevice));
    _gpu_check(cudaMalloc(&d_output, sizeof(size_t) * N * N));

    dim3 numThreadsPerBlock(list_size);
    unsigned int sharedMemorySize = (sizeof(size_t*) * list_size) * (sizeof(size_t) * N * N);

    dreduce<<<1, numThreadsPerBlock, sharedMemorySize>>>(d_list, d_output, pitch);

    _gpu_check(cudaDeviceSynchronize());
    _gpu_check(cudaMemcpy(h_output, d_output, sizeof(size_t) * N * N, cudaMemcpyDeviceToHost));

    _gpu_check(cudaFree(d_output));
    _gpu_check(cudaFree(d_list));
}
