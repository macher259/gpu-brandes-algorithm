# Implementation of Brandes algorithm to compute Betweenness Centrality using CUDA

## Building:
> $ make

## Based on:
Betweenness Centrality on GPUs and Heterogeneous Architectures by Ahmet Erdem Sarıyüce, Kamer Kaya, Erik Saule, Ümit V. Çatalyürek

## Description:
I have implemented algorithm 4 from the paper along with Deg-1 vertices prunning and decomposing graph into connected components. On GPU graph is stored using Stride-CSR representation and virtual vertices are introduced, as in the paper.
I've especially focused on fine-tuning memory layout to ensure the best possible coalescing.

| Dataset  | Time |
|----------|------|
| gowalla  | 76   |
| slashdot | 16   |






