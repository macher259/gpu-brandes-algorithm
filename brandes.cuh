#ifndef CUDA_BRANDES_BRANDES_CUH
#define CUDA_BRANDES_BRANDES_CUH

#include "cuda_helper.cuh"

namespace brandes {
    constexpr int NOT_VISITED = -1;

    struct AlgorithmContext {
        unsigned int *offset{};
        unsigned int *virtual_map{};
        unsigned int *virtual_count{};
        unsigned int *pointers{};
        unsigned int *adjacent{};
        double *bc_values{};
        unsigned int* reach{};

        AlgorithmContext &operator=(AlgorithmContext &&rhs) noexcept {
            offset = rhs.offset;
            rhs.offset = nullptr;

            virtual_map = rhs.virtual_map;
            rhs.virtual_map = nullptr;

            virtual_count = rhs.virtual_count;
            rhs.virtual_count = nullptr;

            pointers = rhs.pointers;
            rhs.pointers = nullptr;

            adjacent = rhs.adjacent;
            rhs.adjacent = nullptr;

            bc_values = rhs.bc_values;
            rhs.bc_values = nullptr;

            reach = rhs.reach;
            rhs.reach = nullptr;
            return *this;
        }
        AlgorithmContext() = default;
        AlgorithmContext(unsigned int real_count, unsigned int virtual_count_, unsigned int edge_count_);
        ~AlgorithmContext();
    };

    struct DeviceAlgorithmContext {
        unsigned int *offset{};
        unsigned int *virtual_map{};
        unsigned int *virtual_count{};
        unsigned int *pointers{};
        unsigned int *adjacent{};
        unsigned int distance_size{};
        int *distance{};
        unsigned int sigma_size{};
        unsigned int *sigma{};
        unsigned int delta_size{};
        double *delta{};
        double *bc_values{};
        bool *should_continue{};
        unsigned int* reach{};

        DeviceAlgorithmContext(AlgorithmContext &host_context, unsigned int real_count, unsigned int virtual_count_, unsigned int edge_count_);
        ~DeviceAlgorithmContext();

        void initializeForSource(unsigned int source) const;
    };
}// namespace brandes

#endif//CUDA_BRANDES_BRANDES_CUH
