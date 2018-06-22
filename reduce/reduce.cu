#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VSIZE 3 // vector size

__device__ int dmin(size_t a, size_t b)
{
    size_t max = ((a+b) + __vabs2(b-a))/2;
    return max - __vabs2(b-a);
}

__global__ void reduce(size_t *input, size_t *output, size_t pitch)
{
    extern __shared__ size_t sdata[];

    // thread id will be used to identify the matrix index
    size_t tid = threadIdx.x;

    // copying all data to shared memory
    for (int k=0; k<VSIZE; k++) {
        size_t *row = &sdata[tid];
        row[k] = input[tid * (pitch / sizeof(size_t)) + k];
    }

    __syncthreads();

    // reducing using interleaved addressing
    for (size_t s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
            for (int idx = 0; idx < VSIZE; ++idx) {
                size_t *row = &sdata[tid];
                size_t *nrow = &sdata[tid + s];
                row[idx] = min(row[idx], nrow[idx]);
            }
        __syncthreads();
    }

    // writing the result
    if (tid == 0) {
        size_t *row = &sdata[0];
        for (int idx = 0; idx < VSIZE; ++idx) {
            output[idx] = row[idx];
        }
    }
}

void _gpu_check(cudaError error)
{
    if (error != cudaSuccess)
    {
        printf("GPU error %d: %s\n", error, cudaGetErrorString(error));
        exit(-1);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: reduce <input_file>\n");
        exit(-1);
    }

    // retrieving the file name
    char *path;
    size_t list_size = 0;
    path = (char*) malloc(sizeof(char) * strlen(argv[1]));
    strcpy(path, argv[1]);

    // creating matrices from file
    FILE *h_file = fopen(path, "r");
    fscanf(h_file, "%d", &list_size);

    // host defs
    size_t h_list[list_size][VSIZE];
    size_t h_output[VSIZE];

    for (size_t i = 0; i < list_size; ++i)
    {
        fscanf(h_file, "%*s");
        for (size_t j = 0; j < VSIZE; ++j)
            fscanf(h_file, "%lu", &h_list[i][j]);
    }
    fclose(h_file);

    // debug
    printf("host matrices\n");
    for (size_t i = 0; i < list_size; ++i)
    {
        for (size_t j=0; j<VSIZE; ++j)
            printf("%lu ", h_list[i][j]);
        printf("\n");
    }

    // device defs
    size_t *d_list;
    size_t *d_output;

    size_t pitch;
    _gpu_check(cudaMallocPitch(&d_list, &pitch, VSIZE * sizeof(size_t), list_size));
    _gpu_check(cudaMemcpy2D(d_list, pitch, h_list, VSIZE * sizeof(size_t), VSIZE * sizeof(size_t), list_size, cudaMemcpyHostToDevice));
    _gpu_check(cudaMalloc(&d_output, sizeof(size_t) * VSIZE));

    dim3 numThreadsPerBlock(list_size);
    unsigned int sharedMemorySize = (sizeof(size_t) * list_size) * (sizeof(size_t) * VSIZE);
    reduce<<<1, numThreadsPerBlock, sharedMemorySize>>>(d_list, d_output, pitch);

    _gpu_check(cudaDeviceSynchronize());
    _gpu_check(cudaMemcpy(h_output, d_output, sizeof(size_t) * VSIZE, cudaMemcpyDeviceToHost));

    _gpu_check(cudaFree(d_output));
    _gpu_check(cudaFree(d_list));

    // print result
    for (size_t j=0; j<VSIZE; ++j)
    {
        printf("%lu ", h_output[j]);
    }
    printf("\n");

    return 0;
}
