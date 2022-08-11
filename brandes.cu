#include "brandes.cuh"

brandes::AlgorithmContext::AlgorithmContext(unsigned int real_count, unsigned int virtual_count_, unsigned int edge_count_) {
    offset = new unsigned int[virtual_count_];
    virtual_map = new unsigned int[virtual_count_];
    virtual_count = new unsigned int[real_count];
    pointers = new unsigned int[real_count + 1];
    adjacent = new unsigned int[edge_count_];
    bc_values = new double[real_count];
    std::fill(bc_values, bc_values + real_count, 0.0);
    reach = new unsigned int[real_count];
    std::fill(reach, reach + real_count, 1);
}

brandes::AlgorithmContext::~AlgorithmContext() {
    delete offset;
    delete virtual_map;
    delete virtual_count;
    delete pointers;
    delete adjacent;
    delete bc_values;
    delete reach;
}

brandes::DeviceAlgorithmContext::DeviceAlgorithmContext(brandes::AlgorithmContext &host_context, unsigned int real_count_, unsigned int virtual_count_, unsigned int edge_count_)
: distance_size(real_count_ * sizeof(unsigned int)), sigma_size(real_count_ * sizeof(unsigned int)), delta_size(real_count_ * sizeof(double))
{
    auto offset_size = virtual_count_ * sizeof(unsigned int);
    HANDLE_ERROR(cudaMalloc(&offset, offset_size));
    HANDLE_ERROR(cudaMemcpy(offset, host_context.offset, offset_size, cudaMemcpyHostToDevice));

    auto vmap_size = virtual_count_ * sizeof(unsigned int);
    HANDLE_ERROR(cudaMalloc(&virtual_map, vmap_size));
    HANDLE_ERROR(cudaMemcpy(virtual_map, host_context.virtual_map, vmap_size, cudaMemcpyHostToDevice));

    auto nvir_size = real_count_ * sizeof(unsigned int);
    HANDLE_ERROR(cudaMalloc(&virtual_count, nvir_size));
    HANDLE_ERROR(cudaMemcpy(virtual_count, host_context.virtual_count, nvir_size, cudaMemcpyHostToDevice));

    auto ptrs_size = (real_count_ + 1) * sizeof(unsigned int);
    HANDLE_ERROR(cudaMalloc(&pointers, ptrs_size));
    HANDLE_ERROR(cudaMemcpy(pointers, host_context.pointers, ptrs_size, cudaMemcpyHostToDevice));

    auto adjs_size = edge_count_ * sizeof(unsigned int);
    HANDLE_ERROR(cudaMalloc(&adjacent, adjs_size));
    HANDLE_ERROR(cudaMemcpy(adjacent, host_context.adjacent, adjs_size, cudaMemcpyHostToDevice));

    auto bc_size = real_count_ * sizeof(double);
    HANDLE_ERROR(cudaMalloc(&bc_values, bc_size));
    HANDLE_ERROR(cudaMemset(bc_values, 0.0, bc_size));

    HANDLE_ERROR(cudaMalloc(&distance, distance_size));
    HANDLE_ERROR(cudaMalloc(&sigma, sigma_size));
    HANDLE_ERROR(cudaMalloc(&delta, delta_size));
    HANDLE_ERROR(cudaMalloc(&should_continue, sizeof(bool)));

    HANDLE_ERROR(cudaMalloc(&reach, sizeof(unsigned int) * real_count_));
    HANDLE_ERROR(cudaMemcpy(reach, host_context.reach, sizeof(unsigned int) * real_count_, cudaMemcpyHostToDevice));
}

void brandes::DeviceAlgorithmContext::initializeForSource(unsigned int source) const {
    HANDLE_ERROR(cudaMemset(distance, NOT_VISITED, distance_size));
    HANDLE_ERROR(cudaMemset(sigma, 0, sigma_size));
    HANDLE_ERROR(cudaMemset(&distance[source], 0, sizeof(unsigned int)));
    HANDLE_ERROR(cudaMemset(&sigma[source], 1, 1));
}

brandes::DeviceAlgorithmContext::~DeviceAlgorithmContext() {
    HANDLE_ERROR(cudaFree(offset));
    HANDLE_ERROR(cudaFree(virtual_map));
    HANDLE_ERROR(cudaFree(virtual_count));
    HANDLE_ERROR(cudaFree(pointers));
    HANDLE_ERROR(cudaFree(adjacent));
    HANDLE_ERROR(cudaFree(bc_values));
    HANDLE_ERROR(cudaFree(distance));
    HANDLE_ERROR(cudaFree(sigma));
    HANDLE_ERROR(cudaFree(delta));
    HANDLE_ERROR(cudaFree(should_continue));
    HANDLE_ERROR(cudaFree(reach));
}
