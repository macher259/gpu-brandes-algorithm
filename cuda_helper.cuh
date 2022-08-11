#ifndef CUDA_BRANDES_CUDA_HELPER_CUH
#define CUDA_BRANDES_CUDA_HELPER_CUH

#include <iostream>

namespace cuda {
#define HANDLE_ERROR(err) (cuda::HandleError(err, __FILE__, __LINE__))

    constexpr unsigned int THREAD_NUMBER = 256;

    __forceinline__ void HandleError(cudaError_t err,
                     const char *file,
                     int line) {
        if (err != cudaSuccess) {
            printf("%s in %s at line %d\n", cudaGetErrorString(err),
                   file, line);
            exit(EXIT_FAILURE);
        }
    }

    __forceinline__ __device__ unsigned int getThreadId() {
        return threadIdx.x + blockIdx.x * blockDim.x;
    }
}// namespace cuda

#endif//CUDA_BRANDES_CUDA_HELPER_CUH
