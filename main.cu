#include "graph.cuh"
#include "stride.cuh"

int main(int argc, char **argv) {
    graph::Graph graph = graph::Graph::importGraph(argv[1]);
    graph.pruneLoneNodes();
    graph.build();

    brandes::StrideBrandesAlgorithm::calculateBetweennessCentrality(graph);

    graph.writeBCValuesToFile(argv[2]);

    return 0;
}
