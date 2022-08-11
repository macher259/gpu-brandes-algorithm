#include "graph.cuh"

constexpr int MDEG = 4;

__forceinline__ int getNumberOfVirtualNodesForNode(int node_deg) {
    return (node_deg + MDEG - 1) / MDEG;
}

int graph::ConnectedComponent::getNodeId(int node) {
    auto it = node_map.find(node);
    if (it == node_map.end()) {
        number_of_original_nodes = std::max(number_of_original_nodes, static_cast<unsigned int>(node) + 1);
        int new_id = static_cast<int>(node_map.size());
        node_map[node] = new_id;
        return new_id;
    } else {
        return it->second;
    }
}
void graph::ConnectedComponent::addDirectedEdge(int node_u, int node_v) {
    if (node_u != node_v) {
        int u_id = getNodeId(node_u);
        int v_id = getNodeId(node_v);

        adjacency_list[u_id].insert(v_id);
        number_of_edges++;
    }
}
void graph::ConnectedComponent::build() {
    calculateNumberOfVirtualNodes();
    context = brandes::AlgorithmContext(getNumberOfRealNodes(), getNumberOfVirtualNodes(), getNumberOfEdges());

    int offset_id = 0;
    int adjacent_id = 0;
    for (const auto &adj: adjacency_list) {
        context.virtual_count[adj.first] = getNumberOfVirtualNodesForNode(static_cast<int>(adj.second.size()));

        for (int i = 0; i < context.virtual_count[adj.first]; ++i) {
            context.offset[offset_id] = i;
            context.virtual_map[offset_id] = adj.first;
            offset_id++;
        }

        context.pointers[adj.first] = adjacent_id;
        for (auto it: adjacency_list[adj.first]) {
            context.adjacent[adjacent_id] = it;
            adjacent_id++;
        }
        context.pointers[getNumberOfRealNodes()] = adjacent_id;
    }
}
void graph::ConnectedComponent::calculateNumberOfVirtualNodes() {
    number_of_virtual_nodes = 0;
    for (const auto &adj: adjacency_list)
        number_of_virtual_nodes += getNumberOfVirtualNodesForNode(static_cast<int>(adj.second.size()));
}

graph::Graph::Graph(const std::string &filename) {
    std::ifstream file(filename);
    Graph g;
    int u;
    int v;
    while (file >> u >> v)
        addEdge(u, v);
}

int graph::Graph::getNodeId(int node) {
    auto it = node_map.find(node);
    if (it == node_to_component_map.end()) {
        number_of_original_nodes = std::max(number_of_original_nodes, static_cast<unsigned int>(node) + 1);
        int new_id = static_cast<int>(node_map.size());
        node_map[node] = new_id;
        return new_id;
    } else {
        return it->second;
    }
}

void graph::Graph::addEdge(int node_u, int node_v) {
    if (node_u != node_v) {
        int u_id = getNodeId(node_u);
        int v_id = getNodeId(node_v);

        adjacency_list[u_id].insert(v_id);
        adjacency_list[v_id].insert(u_id);
        number_of_edges++;
    }
}
void graph::Graph::build() {
    int number_of_connected_components = discoverConnectedComponents();
    createConnectedComponents(number_of_connected_components);

    for (auto &component: connected_components) {
        component.build();
    }

    int v;
    for (auto &it: node_to_component_map) {
        v = it.first;
        connected_components[node_to_component_map[v]].set_reach(v, pruned_reach[v]);
    }
}

void graph::Graph::createConnectedComponents(int number_of_components) {
    connected_components = std::vector<ConnectedComponent>(number_of_components);

    for (auto &adj: adjacency_list) {
        int component = node_to_component_map[adj.first];
        for (int neighbor: adjacency_list[adj.first]) {
            connected_components[component].addDirectedEdge(adj.first, neighbor);
        }
    }
}

int graph::Graph::discoverConnectedComponents() {
    int number_of_components = 0;

    for (auto &adj: adjacency_list) {
        if (node_to_component_map.find(adj.first) == node_to_component_map.end()) {
            markComponent(adj.first, number_of_components);
            number_of_components++;
        }
    }
    return number_of_components;
}

int graph::Graph::markComponent(int node, int component_id) {
    std::queue<int> q;
    q.push(node);
    int count = 0;

    while (!q.empty()) {
        ++count;
        node = q.front();
        q.pop();

        for (int neighbor: adjacency_list[node]) {
            if (node_to_component_map.find(neighbor) == node_to_component_map.end()) {
                node_to_component_map[neighbor] = component_id;
                q.push(neighbor);
            }
        }
    }
    return count;
}
void graph::Graph::writeBCValuesToFile(const std::string &filename) {
    std::ofstream file(filename);

    for (int node = 0; node < number_of_original_nodes; node++) {
        double bc_value = 0;
        if (node_map.find(node) != node_map.end()) {
            int node_id = node_map[node];
            bc_value = pruned_bc[node_id];
            if (!isNodePruned(node_id))
                bc_value += connected_components[node_to_component_map[node_id]].getBCValueForNode(node_id);
        }
        file << bc_value << std::endl;
    }
}
graph::Graph graph::Graph::importGraph(const std::string &filename) {
    std::ifstream file(filename);
    Graph g;
    int u;
    int v;
    while (file >> u >> v)
        g.addEdge(u, v);
    return g;
}
void graph::Graph::pruneLoneNodes() {
    pruned_bc.assign(node_map.size(), 0.0);
    pruned_reach.assign(node_map.size(), 1);

    auto cc_to_size = pruneDiscoverConnectedComponents();
    auto deg_to_nodes = buildDegToNodesMap();

    int64_t N;
    int total_pruned = 0, pruned = 1;
    while (pruned > 0) {
        pruned = 0;
        std::unordered_set<int> next_iteration_deg1;

        for (auto tr: deg_to_nodes[1]) {
            N = cc_to_size[node_to_component_map[tr]] - 1;
            pruned_bc[tr] += static_cast<double>(pruned_reach[tr] - 1) * static_cast<double>(N - pruned_reach[tr]);

            for (auto neigh: adjacency_list[tr]) {
                adjacency_list[neigh].erase(tr);

                size_t new_deg = adjacency_list[neigh].size();
                deg_to_nodes[new_deg + 1].erase(neigh);
                if (new_deg == 1) {
                    next_iteration_deg1.insert(neigh);
                } else {
                    deg_to_nodes[new_deg].insert(neigh);
                }

                pruned_reach[neigh] += pruned_reach[tr];
                pruned_bc[neigh] += static_cast<double>(pruned_reach[tr] * (N - pruned_reach[tr] - 1));
            }

            adjacency_list.erase(tr);
            total_pruned++;
            pruned++;
        }

        deg_to_nodes[1] = std::move(next_iteration_deg1);
    }
    node_to_component_map.clear();
}

std::unordered_map<int, std::unordered_set<int>> graph::Graph::buildDegToNodesMap() {
    std::unordered_map<int, std::unordered_set<int>> deg_to_nodes;
    for (const auto &it: adjacency_list) {
        deg_to_nodes[it.second.size()].insert(it.first);
    }
    return deg_to_nodes;
}

std::unordered_map<int, int> graph::Graph::pruneDiscoverConnectedComponents() {
    std::unordered_map<int, int> cc_to_size;
    int cc = 0;
    for (auto &it_adj: adjacency_list) {
        if (node_to_component_map.find(it_adj.first) == node_to_component_map.end()) {
            cc_to_size[cc] = markComponent(it_adj.first, cc);
            ++cc;
        }
    }
    return cc_to_size;
}
