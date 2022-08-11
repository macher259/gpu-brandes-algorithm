#ifndef CUDA_BRANDES_GRAPH_CUH
#define CUDA_BRANDES_GRAPH_CUH

#include "brandes.cuh"
#include <fstream>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace graph {
    using AdjacencyList = std::map<int, std::set<int>>;

    struct ConnectedComponent {
        brandes::AlgorithmContext context;
        unsigned int number_of_edges{};
        unsigned int number_of_virtual_nodes{};
        unsigned int number_of_original_nodes{};

        std::unordered_map<int, int> node_map;
        AdjacencyList adjacency_list;

        int getNodeId(int node);
        void addDirectedEdge(int node_u, int node_v);
        void calculateNumberOfVirtualNodes();
        void build();
        unsigned int getNumberOfRealNodes() const {
            return node_map.size();
        }
        unsigned int getNumberOfVirtualNodes() const {
            return number_of_virtual_nodes;
        }
        unsigned int getNumberOfEdges() const {
            return number_of_edges;
        }
        double getBCValueForNode(int node) {
            return context.bc_values[node_map[node]];
        }

        void set_reach(int node, unsigned int val) {
            context.reach[node_map[node]] = val;
        }
    };

    struct Graph {
        unsigned int number_of_edges{};
        unsigned int number_of_original_nodes{};

        std::unordered_map<int, int> node_map{};
        AdjacencyList adjacency_list{};
        std::unordered_map<int, int> node_to_component_map{};
        std::vector<ConnectedComponent> connected_components{};
        std::vector<double> pruned_bc{};
        std::vector<unsigned int> pruned_reach;

        Graph() = default;
        explicit Graph(const std::string &filename);
        static Graph importGraph(const std::string &filename);
        int getNodeId(int node);
        void addEdge(int node_u, int node_v);
        void build();
        void createConnectedComponents(int number_of_components);
        int discoverConnectedComponents();
        int markComponent(int node, int component_id);
        void writeBCValuesToFile(const std::string &filename);
        void pruneLoneNodes();
        std::unordered_map<int, std::unordered_set<int>> buildDegToNodesMap();
        std::unordered_map<int, int> pruneDiscoverConnectedComponents();

        bool isNodePruned(int node_id) {
            return adjacency_list.find(node_id) == adjacency_list.end();
        }
    };
}// namespace graph

#endif//CUDA_BRANDES_GRAPH_CUH
