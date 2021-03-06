#include <stdio.h>

__global__ void square(float *d_out, float *d_in)
{
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f*f;
}

int main(int argc, char *argv[])
{
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // inicializando o array de input no host (prefixo h-host)
    float h_in[ARRAY_SIZE];
    for (int i=0; i<ARRAY_SIZE; i++)
        h_in[i] = float(i);

    // array de output a ser lido no host
    float h_output[ARRAY_SIZE];

    // ponteiros para serem usados no device (prefixo d-device)
    float *d_in;
    float *d_out;

    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);

    // destination, source, number of bytes, 
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // <<< >>> lauch operator
    // isso indica, execute o kernel square usando um bloco de 64 elementos
    // 64 threads
    square<<<1, ARRAY_SIZE>>>(d_out, d_in);
    cudaMemcpy(h_output, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    for (int i=0; i<ARRAY_SIZE; i++)
    {
        printf("%f", h_output[i]);
        printf(((i % 4) != 3) ? "\t" : "\n");
    }
    
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

