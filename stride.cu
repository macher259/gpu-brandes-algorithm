#include "brandes.cuh"
#include "stride.cuh"

__forceinline__ __device__ unsigned int calculateStride() {
    return blockDim.x * gridDim.x;
}

__global__ void forwardPass(int level, bool *should_continue, int *distance, unsigned int *sigma, const unsigned int *offset, const unsigned int *virtual_map, const unsigned int *virtual_count, const unsigned int *pointers, const unsigned int *adjacent, unsigned int number_of_virtual_nodes) {
    auto id = cuda::getThreadId();
    auto stride = calculateStride();

    for (auto virtual_node = id; virtual_node < number_of_virtual_nodes; virtual_node += stride) {
        auto node = virtual_map[virtual_node];

        if (distance[node] == level) {
            auto start = pointers[node] + offset[virtual_node];
            auto end = pointers[node + 1];
            auto neighbor_stride = virtual_count[node];

            for (auto neighbor_id = start; neighbor_id < end; neighbor_id += neighbor_stride) {
                auto neighbor = adjacent[neighbor_id];

                if (distance[neighbor] == brandes::NOT_VISITED) {
                    distance[neighbor] = level + 1;
                    *should_continue = true;
                }

                if (distance[neighbor] == level + 1) {
                    atomicAdd(&sigma[neighbor], sigma[node]);
                }
            }
        }
    }
}

__global__ void backwardPass(int level, const int *distance, double *delta, const unsigned int *offset, const unsigned int *virtual_map, const unsigned int *virtual_count, const unsigned int *pointers, const unsigned int *adjacent, unsigned int number_of_virtual_nodes) {
    auto id = cuda::getThreadId();
    auto stride = calculateStride();

    for (unsigned int virtual_node = id; virtual_node < number_of_virtual_nodes; virtual_node += stride) {
        auto node = virtual_map[virtual_node];

        if (distance[node] == level) {
            double thread_sum = 0.0;
            auto start = pointers[node] + offset[virtual_node];
            auto end = pointers[node + 1];
            auto neighbor_stride = virtual_count[node];

            for (auto neighbor_id = start; neighbor_id < end; neighbor_id += neighbor_stride) {
                auto neighbor = adjacent[neighbor_id];

                if (distance[neighbor] == level + 1) {
                    thread_sum += delta[neighbor];
                }
            }
            atomicAdd(&delta[node], thread_sum);
        }
    }
}

__global__ void updateDeltas(unsigned int number_of_nodes, double *delta, const unsigned int *sigma, unsigned int* reach) {
    auto id = cuda::getThreadId();
    auto stride = calculateStride();

    for (auto node = id; node < number_of_nodes; node += stride) {
        if (sigma[node] != 0) {
            delta[node] = static_cast<double>(reach[node]) / sigma[node];
        }
    }
}
__global__ void updateBCValues(unsigned int source, unsigned int number_of_nodes, double *bc_values, const double *delta, const unsigned int *sigma, unsigned int* reach) {
    auto id = cuda::getThreadId();
    auto stride = calculateStride();

    for (auto node = id; node < number_of_nodes; node += stride) {
        if (node != source) {
            double update = (delta[node] * sigma[node] - 1.0) * static_cast<double>(reach[source]);

            if (update > 0) {
                bc_values[node] += update;
            }
        }
    }
}

__forceinline__ void startNewBenchmark(cudaEvent_t &start, cudaEvent_t &stop) {
    HANDLE_ERROR(cudaEventRecord(start, 0));
    HANDLE_ERROR(cudaEventSynchronize(start));
}

__forceinline__ float endBenchmark(cudaEvent_t &start, cudaEvent_t &stop) {
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    return elapsedTime;
}

void brandes::StrideBrandesAlgorithm::calculateBetweennessCentrality(graph::Graph &graph) {
    double memory_transfer_time = 0;
    double kernel_execution_time = 0;

    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));
    for (auto &component: graph.connected_components) {
        DeviceAlgorithmContext gpu_context(component.context, component.getNumberOfRealNodes(), component.getNumberOfVirtualNodes(), component.getNumberOfEdges());

        for (unsigned int source = 0; source < component.getNumberOfRealNodes(); ++source) {
            startNewBenchmark(start, stop);
            gpu_context.initializeForSource(source);
            memory_transfer_time += endBenchmark(start, stop);

            int level = 0;
            bool host_should_continue = true;

            while (host_should_continue) {
                HANDLE_ERROR(cudaMemset(gpu_context.should_continue, false, sizeof(bool)));

                startNewBenchmark(start, stop);
                forwardPass<<<calculateGridDimensions(component.getNumberOfVirtualNodes()), calculateBlockDimensions()>>>(
                        level,
                        gpu_context.should_continue,
                        gpu_context.distance,
                        gpu_context.sigma,
                        gpu_context.offset,
                        gpu_context.virtual_map,
                        gpu_context.virtual_count,
                        gpu_context.pointers,
                        gpu_context.adjacent,
                        component.getNumberOfVirtualNodes());
                kernel_execution_time += endBenchmark(start, stop);

                HANDLE_ERROR(cudaMemcpy(&host_should_continue, gpu_context.should_continue, sizeof(bool), cudaMemcpyDeviceToHost));
                ++level;
            }

            startNewBenchmark(start, stop);

            updateDeltas<<<calculateGridDimensions(component.getNumberOfRealNodes()), calculateBlockDimensions()>>>(component.getNumberOfRealNodes(), gpu_context.delta, gpu_context.sigma, gpu_context.reach);
            while (level > 1) {
                --level;
                backwardPass<<<calculateGridDimensions(component.getNumberOfVirtualNodes()), calculateBlockDimensions()>>>(
                        level,
                        gpu_context.distance,
                        gpu_context.delta,
                        gpu_context.offset,
                        gpu_context.virtual_map,
                        gpu_context.virtual_count,
                        gpu_context.pointers,
                        gpu_context.adjacent,
                        component.getNumberOfVirtualNodes());
            }

            updateBCValues<<<calculateGridDimensions(component.getNumberOfRealNodes()), calculateBlockDimensions()>>>(
                    source, component.getNumberOfRealNodes(), gpu_context.bc_values, gpu_context.delta, gpu_context.sigma, gpu_context.reach);
        }
        kernel_execution_time += endBenchmark(start, stop);
        startNewBenchmark(start, stop);

        HANDLE_ERROR(cudaMemcpy(component.context.bc_values, gpu_context.bc_values, component.getNumberOfRealNodes() * sizeof(double), cudaMemcpyDeviceToHost));
        memory_transfer_time += endBenchmark(start, stop);
    }

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    std::cerr << std::round(kernel_execution_time) << std::endl;
    std::cerr << std::round(memory_transfer_time + kernel_execution_time) << std::endl;
}
