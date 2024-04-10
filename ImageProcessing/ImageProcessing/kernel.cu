
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>

#include "ImageParser.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void multiplyKernel(int* c, const int* a, const int* b, int size_of_array)
{

    //set thread ID
    unsigned int cuda_education_thread_id = threadIdx.x;

    unsigned int linear_id_of_thread = blockIdx.x * blockDim.x + threadIdx.x;

    //convert global data pointer to the local pointer of this block
    int* local_data_of_block = c + blockIdx.x * blockDim.x;

    if (linear_id_of_thread >= size_of_array) return;  //make sure the number of threads that are processing the array is not greater than the size of the array 



    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

int main()
{
    ImageParser parser = ImageParser("1660578195.bmp");
    
    const int arraySize = 1024;
    std::cout << arraySize << std::endl;
    int a[arraySize] = { 2 };
    std::fill_n(a, arraySize, 2);
    unsigned char* imagearr = parser.readBMP();
    int imageInt[arraySize] = { 0 };
    for (int i = 0; i < arraySize; i++) {
        imageInt[i] = (int)imagearr[i];
    }
    int c[arraySize] = { 0 };

    int num_array_each_block = 256;

    int val = parser.readBMPToArray()[575][1023][0];

    printf("image val in red channel at 575 1023: %d\n", val);

    int val2 = parser.turnGreyScale(parser.readBMPToArray())[575][1023];

    printf("image val in greyscale channel at 575 1023: %d\n", val2);

    // Creates a BLOCK variable
    dim3 BLOCK(num_array_each_block, 1);

    // calculation of number of blocks needed based in threads.
    dim3 cuda_education_grid((arraySize + BLOCK.x - 1) / BLOCK.x, 1);

    printf("cuda_education_grid %d | BLOCK %d\n", cuda_education_grid.x, BLOCK.x);

    //MEMORY MANAGEMENT ON HOST

    size_t int_bytes = arraySize * sizeof(int);

    int *host_data_send_to_device = (int*)malloc(int_bytes);

    int* host_data_received_from_device = (int*)malloc(cuda_education_grid.x * sizeof(int));

    //MEMORY MANAGMENT ON HOST END

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    //cudaSetDevice(0);

    // Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, imageInt, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4],c[5]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

    

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

    

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel <<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
