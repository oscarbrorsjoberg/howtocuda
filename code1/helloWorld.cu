#include <assert.h>
#include <stdio.h>
#include <cuda_runtime_api.h>

#define cudaCheckError(code) {                            \
    if((code) != cudaSuccess) {                           \
        fprintf(stderr, "Cuda failure %s:%d: '%s'  \n",   \
                __FILE__, __LINE__,                       \
                cudaGetErrorString(code));                \
  }                                                       \
} 

void add_loop(float *dest, int n_elts, const float *a, const float *b)
{
    for(int i = 0; i < n_elts; i++)
        dest[i] = a[i] + b[i];
}

__global__ void add_kernel(float *dest, int n_elts, const float *a, const float *b)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= n_elts)
        return;

    dest[index] = a[index] + b[index];
}


int main(){


    const int AL = 100;


    // generate some data on the host
    float host_array_a[AL];
    float host_array_b[AL];
    float host_array_dest[AL];

    for(int i = 0; i < AL; i++ ){
        host_array_a[i] = 2 * i;
        host_array_b[i] = 2 * i + 1;
    }

    // Allocate device memory
    float *device_array_a, *device_array_b, *device_array_dest;

    cudaCheckError(cudaMalloc(&device_array_a,    sizeof(host_array_a)));
    cudaCheckError(cudaMalloc(&device_array_b,    sizeof(host_array_b))); 
    cudaCheckError(cudaMalloc(&device_array_dest, sizeof(host_array_dest)));

    cudaCheckError(cudaMemcpy(device_array_a, host_array_a, sizeof(host_array_a),
            cudaMemcpyHostToDevice));

    cudaCheckError(cudaMemcpy(device_array_b, host_array_b, sizeof(host_array_b),
            cudaMemcpyHostToDevice));
    
    const int BS = 128;
    int n_blocks = (AL + BS - 1 ) / BS;

    add_kernel<<<BS, n_blocks>>>(device_array_dest, AL, 
                                    device_array_a, device_array_b);

    // Meanwhile, add arrays on the host, for comparison
    add_loop(host_array_dest, AL, host_array_a, host_array_b);


    // Copy result back to the host and compare
    float host_array_tmp[AL];


    cudaCheckError(cudaMemcpy(host_array_tmp, device_array_dest, 
                    sizeof(host_array_tmp),
                    cudaMemcpyDeviceToHost)
            );

    for(int i = 0 ; i < AL; i++ ){
        /* assert(host_array_tmp[i] == host_array_b[i]); */
        printf("%g + %g = %g\n", host_array_a[i], host_array_b[i], host_array_tmp[i]);
    }

    return 0;
}
