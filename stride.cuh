#ifndef CUDA_BRANDES_STRIDE_CUH
#define CUDA_BRANDES_STRIDE_CUH

#include "graph.cuh"
namespace brandes {
    struct StrideBrandesAlgorithm {
        static void calculateBetweennessCentrality(graph::Graph &graph);
    };

    __forceinline__ dim3 calculateGridDimensions(unsigned int number_of_nodes) {
        unsigned int number_of_blocks = (number_of_nodes + cuda::THREAD_NUMBER - 1) / cuda::THREAD_NUMBER;
        return {number_of_blocks, 1, 1};
    }

    __forceinline__ dim3 calculateBlockDimensions() {
        return {cuda::THREAD_NUMBER, 1, 1};
    }
}// namespace brandes

#endif//CUDA_BRANDES_STRIDE_CUH
