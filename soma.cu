#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

__global__ 
void get(int *output)
{
    *output = 10;;
}

int main(int argc, char *argv[])
{
    int *h_num;
    int *d_num;    
    cudaError_t cudaerr;

    h_num = (int*)malloc(sizeof(int));

    cudaerr = cudaMalloc((void **) &d_num, sizeof(int));
    if (cudaerr != cudaSuccess)
        printf("nao pode alocar memoria no device\n");

    get<<<1, 1>>>(d_num);

    cudaerr = cudaMemcpy(h_num, d_num, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaerr != cudaSuccess)
        printf("nao pode copiar memoria\n");

    printf("%d\n", *h_num);

    cudaFree(d_num);
    return 0;
}

