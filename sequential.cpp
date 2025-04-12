#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <random>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <chrono>
// #include <omp.h>
#include "/opt/homebrew/Cellar/libomp/20.1.2/include/omp.h"

class Graph {
private:
    std::unordered_map<int, std::vector<int>> adjList;
    int numNodes;
    int numEdges;

public:
    Graph() : numNodes(0), numEdges(0) {}

    void addEdge(int from, int to) {
        adjList[from].push_back(to);
        adjList[to].push_back(from);
        
        numNodes = std::max(numNodes, std::max(from, to) + 1);
        numEdges++;
    }

    void loadFromFileGML(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        std::string line;
        bool inGraph = false;
        bool inNode = false;
        bool inEdge = false;
        int currentNodeId = -1;
        int sourceNode = -1;
        int targetNode = -1;
        
        while (std::getline(file, line)) {
            // Trim leading and trailing whitespace
            line.erase(0, line.find_first_not_of(" \t"));
            if (line.empty()) continue; // Skip empty lines
            
            // Parse GML format
            if (line == "graph") {
                continue;
            } else if (line == "[") {
                if (!inGraph) {
                    inGraph = true;
                    continue;
                } else if (!inNode && !inEdge && line == "[") {
                    // This is the beginning of a node or edge block
                    continue;
                }
            } else if (line == "]") {
                if (inNode) {
                    inNode = false;
                    currentNodeId = -1;
                } else if (inEdge) {
                    inEdge = false;
                    if (sourceNode != -1 && targetNode != -1) {
                        addEdge(sourceNode, targetNode);
                        sourceNode = -1;
                        targetNode = -1;
                    }
                } else if (inGraph) {
                    inGraph = false;
                }
                continue;
            }
            
            if (!inGraph) continue;
            
            // Check for node definition
            if (line == "node") {
                inNode = true;
                continue;
            }
            
            // Check for edge definition
            if (line == "edge") {
                inEdge = true;
                continue;
            }
            
            // Parse node ID
            if (inNode && line.find("id ") == 0) {
                std::istringstream iss(line.substr(3));
                iss >> currentNodeId;
                continue;
            }
            
            // Parse edge source
            if (inEdge && line.find("source ") == 0) {
                std::istringstream iss(line.substr(7));
                iss >> sourceNode;
                continue;
            }
            
            // Parse edge target
            if (inEdge && line.find("target ") == 0) {
                std::istringstream iss(line.substr(7));
                iss >> targetNode;
                continue;
            }
        }
        
        file.close();
        std::cout << "Graph loaded successfully from GML!" << std::endl;
        std::cout << "Nodes: " << numNodes << ", Edges: " << numEdges / 2 << std::endl;
    }

    void loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        std::string line;
        bool headerProcessed = false;

        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') {
                // Try to extract metadata from comments
                if (line.find("Nodes:") != std::string::npos && 
                    line.find("Edges:") != std::string::npos) {
                    // Parse the line to get number of nodes and edges
                    std::istringstream iss(line);
                    std::string token;
                    iss >> token; // Skip "# DBLP"
                    iss >> token >> numNodes; // "Nodes:" and number
                    iss >> token >> numEdges; // "Edges:" and number
                }
                continue;
            }

            // Skip the header line
            if (!headerProcessed) {
                headerProcessed = true;
                continue;
            }

            // Parse edge
            std::istringstream iss(line);
            int from, to;
            if (iss >> from >> to) {
                addEdge(from, to);
            }
        }

        file.close();
        std::cout << "Graph loaded successfully!" << std::endl;
        std::cout << "Nodes: " << numNodes << ", Edges: " << numEdges / 2 << std::endl;
        // Note: numEdges is divided by 2 because we count each edge twice (once in each direction)
    }

    void printGraph() {
        std::cout << "Graph adjacency list:" << std::endl;
        for (const auto& pair : adjList) {
            std::cout << pair.first << " -> ";
            for (int neighbor : pair.second) {
                std::cout << neighbor << " ";
            }
            std::cout << std::endl;
        }
    }

    int getNumNodes() const { return numNodes; }
    int getNumEdges() const { return numEdges / 2; }
    
    const std::unordered_map<int, std::vector<int>>& getAdjList() const {
        return adjList;
    }
    
    // Get all node IDs in the graph
    std::vector<int> getNodeIDs() const {
        std::vector<int> nodes;
        for (const auto& pair : adjList) {
            nodes.push_back(pair.first);
        }
        return nodes;
    }

    // Additional methods for graph analysis can be added here
};

// Edge representation for pheromone storage
struct Edge {
    int from;
    int to;
    
    Edge(int f, int t) : from(std::min(f, t)), to(std::max(f, t)) {}
    
    bool operator==(const Edge& other) const {
        return from == other.from && to == other.to;
    }
};

// Hash function for Edge
namespace std {
    template <>
    struct hash<Edge> {
        size_t operator()(const Edge& e) const {
            return hash<int>()(e.from) ^ (hash<int>()(e.to) << 1);
        }
    };
}



class ACOCommunityDetection {
private:
    const Graph& graph;
    int num_ants;
    int max_iterations;
    double alpha;           // Pheromone influence
    double beta;            // Heuristic influence
    double rho;             // Evaporation rate
    double initial_pheromone;
    double similarity_threshold; // Threshold for community membership
    
    // Pheromone on edges
    std::unordered_map<Edge, double> pheromones;

    // Store original edge weights (all 1.0 for unweighted graph)
    std::unordered_map<Edge, double> edge_weights;

    // Random number generator
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;

    // Initialize pheromones on all edges
    void initializePheromones() {
        const auto& adj_list = graph.getAdjList();
        
        for (const auto& pair : adj_list) {
            int from = pair.first;
            for (int to : pair.second) {
                if (from < to) { // Only process each edge once
                    Edge edge(from, to);
                    pheromones[edge] = initial_pheromone;
                    edge_weights[edge] = 1.0; // Initial weight (could be modified for weighted graphs)
                }
            }
        }
    }

    // Perform a random walk with an ant
    std::vector<int> antRandomWalk(int start_node, int steps) {
        const auto& adj_list = graph.getAdjList();
        std::vector<int> path;
        int current_node = start_node;
        path.push_back(current_node);
        
        for (int step = 0; step < steps; step++) {
            // Get neighbors of current node
            if (adj_list.find(current_node) == adj_list.end() || adj_list.at(current_node).empty()) {
                break; // No neighbors to visit
            }
            
            const auto& neighbors = adj_list.at(current_node);
            
            // Calculate probabilities based on pheromone levels and heuristic information
            std::vector<double> probabilities;
            double total = 0.0;
            
            for (int neighbor : neighbors) {
                Edge edge(current_node, neighbor);
                double pheromone = pheromones[edge];
                double weight = edge_weights[edge];
                double probability = std::pow(pheromone, alpha) * std::pow(weight, beta);
                probabilities.push_back(probability);
                total += probability;
            }
            
            // Normalize probabilities
            if (total > 0) {
                for (auto& p : probabilities) {
                    p /= total;
                }
            } else {
                // If all probabilities are zero, use uniform distribution
                for (auto& p : probabilities) {
                    p = 1.0 / neighbors.size();
                }
            }
            
            // Select next node using roulette wheel selection
            double r = dist(rng);
            double cumulative = 0.0;
            int next_node = neighbors[0]; // Default
            
            for (size_t i = 0; i < neighbors.size(); i++) {
                cumulative += probabilities[i];
                if (r <= cumulative) {
                    next_node = neighbors[i];
                    break;
                }
            }
            
            current_node = next_node;
            path.push_back(current_node);
        }
        
        return path;
    }

    // Update pheromones based on ant walks
    void updatePheromones(const std::vector<std::vector<int>>& ant_paths) {
        // Evaporation
        for (auto& pair : pheromones) {
            pair.second *= (1.0 - rho);
        }
        
        // Deposit new pheromones based on paths
        for (const auto& path : ant_paths) {
            // Calculate path quality (e.g., based on how much it stays within communities)
            // For now we use a simple reinforcement approach where each visited edge gets a fixed deposit
            double deposit = 1.0 / path.size();
            
            for (size_t i = 0; i < path.size() - 1; i++) {
                Edge edge(path[i], path[i + 1]);
                pheromones[edge] += deposit;
            }
        }
    }

    // Extract communities from pheromone-weighted graph
    std::unordered_map<int, std::set<int>> extractCommunities() {
        std::unordered_map<int, std::set<int>> communities;
        std::unordered_set<int> processed_nodes;
        const auto& adj_list = graph.getAdjList();
        
        // Process each node that hasn't been assigned to a community yet
        for (const auto& pair : adj_list) {
            int node = pair.first;
            
            if (processed_nodes.find(node) != processed_nodes.end()) {
                continue; // Skip if already processed
            }
            
            // Start a new community with this node
            int community_id = node;
            std::set<int> community;
            community.insert(node);
            processed_nodes.insert(node);
            
            // Queue for nodes to be considered
            std::vector<int> queue;
            queue.push_back(node);
            
            // Process queue
            while (!queue.empty()) {
                int current = queue.front();
                queue.erase(queue.begin());
                
                // Check all neighbors
                if (adj_list.find(current) != adj_list.end()) {
                    for (int neighbor : adj_list.at(current)) {
                        if (processed_nodes.find(neighbor) != processed_nodes.end()) {
                            continue; // Skip if already processed
                        }
                        
                        // Calculate similarity to community
                        double similarity = calculateSimilarity(neighbor, community);
                        
                        // Add to community if similarity is above threshold
                        if (similarity >= similarity_threshold) {
                            community.insert(neighbor);
                            processed_nodes.insert(neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
            
            // Store community
            if (!community.empty()) {
                communities[community_id] = community;
            }
        }
        
        return communities;
    }

    // Calculate similarity of a node to a community based on pheromone levels
    double calculateSimilarity(int node, const std::set<int>& community) {
        const auto& adj_list = graph.getAdjList();
        
        if (adj_list.find(node) == adj_list.end()) {
            return 0.0; // No connections
        }
        
        double total_pheromone = 0.0;
        int connections = 0;
        
        // Sum pheromones on edges to community members
        for (int neighbor : adj_list.at(node)) {
            if (community.find(neighbor) != community.end()) {
                Edge edge(node, neighbor);
                total_pheromone += pheromones[edge];
                connections++;
            }
        }
        
        // If no connections to community, similarity is 0
        if (connections == 0) {
            return 0.0;
        }
        
        // Average pheromone level
        return total_pheromone / connections;
    }

    // Calculate modularity of community structure
    double calculateModularity(const std::unordered_map<int, std::set<int>>& communities) {
        const auto& adj_list = graph.getAdjList();
        double m = graph.getNumEdges(); // Total number of edges
        double q = 0.0;
        
        // Create mapping from node to community
        std::unordered_map<int, int> node_community;
        for (const auto& pair : communities) {
            int community_id = pair.first;
            for (int node : pair.second) {
                node_community[node] = community_id;
            }
        }
        
        // Calculate degrees of each node
        std::unordered_map<int, double> degrees;
        for (const auto& pair : adj_list) {
            int node = pair.first;
            degrees[node] = pair.second.size();
        }
        
        // Calculate modularity
        for (const auto& pair : adj_list) {
            int i = pair.first;
            
            if (node_community.find(i) == node_community.end()) {
                continue; // Skip nodes not assigned to communities
            }
            
            int c_i = node_community[i];
            
            for (int j : pair.second) {
                if (node_community.find(j) == node_community.end()) {
                    continue;
                }
                
                int c_j = node_community[j];
                
                if (c_i == c_j) {
                    q += 1.0 - (degrees[i] * degrees[j]) / (2.0 * m);
                }
            }
        }
        
        q /= (2.0 * m);
        return q;
    }

public:
    ACOCommunityDetection(const Graph& g, int ants = 20, int iterations = 100,
                              double a = 1.0, double b = 0.5, double r = 0.1, 
                              double init_pheromone = 1.0, double sim_threshold = 0.5)
        : graph(g), num_ants(ants), max_iterations(iterations),
          alpha(a), beta(b), rho(r), initial_pheromone(init_pheromone),
          similarity_threshold(sim_threshold),
          rng(std::random_device{}()), dist(0.0, 1.0) {
        initializePheromones();
    }

    // Sequential version
    std::unordered_map<int, std::set<int>> run() {
        const auto& node_ids = graph.getNodeIDs();
        int walk_steps = std::min(100, (int)(graph.getNumNodes() / 10));
        
        std::cout << "Running ACO with " << num_ants << " ants for " << max_iterations 
                  << " iterations, walk length: " << walk_steps << std::endl;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            std::vector<std::vector<int>> ant_paths;
            
            // Each ant performs a random walk
            for (int ant = 0; ant < num_ants; ant++) {
                // Select random starting node
                int start_node = node_ids[dist(rng) * node_ids.size()];
                
                // Perform random walk
                std::vector<int> path = antRandomWalk(start_node, walk_steps);
                ant_paths.push_back(path);
            }
            
            // Update pheromones
            updatePheromones(ant_paths);
            
            if (iter % 10 == 0 || iter == max_iterations - 1) {
                // Extract communities and calculate modularity periodically
                auto communities = extractCommunities();
                double modularity = calculateModularity(communities);
                std::cout << "Iteration " << iter << ": " << communities.size()
                          << " communities, modularity = " << modularity << std::endl;
            }
        }
        
        // Final community extraction
        auto communities = extractCommunities();
        double modularity = calculateModularity(communities);
        std::cout << "Final result: " << communities.size() 
                  << " communities, modularity = " << modularity << std::endl;
        
        return communities;
    }
};

// Utility function to print communities
void printCommunities(const std::unordered_map<int, std::set<int>>& communities) {
    // Create a vector of communities sorted by size
    std::vector<std::pair<int, std::set<int>>> sorted_communities;
    for (const auto& pair : communities) {
        sorted_communities.push_back(pair);
    }
    
    std::sort(sorted_communities.begin(), sorted_communities.end(),
              [](const auto& a, const auto& b) { return a.second.size() > b.second.size(); });
    
    std::cout << "Detected " << communities.size() << " communities:" << std::endl;
    
    // Print top 10 communities
    int count = 0;
    for (const auto& pair : sorted_communities) {
        std::cout << "Community " << pair.first << " (size: " << pair.second.size() << "): ";
        
        // Print first 10 nodes in each community
        int node_count = 0;
        for (int node : pair.second) {
            if (node_count++ < 10) {
                std::cout << node << " ";
            } else {
                break;
            }
        }
        
        if (pair.second.size() > 10) {
            std::cout << "...";
        }
        
        std::cout << std::endl;
        
        if (++count >= 10 && communities.size() > 10) {
            std::cout << "... (" << (communities.size() - 10) << " more communities)" << std::endl;
            break;
        }
    }
}

// Save community structure to file
void saveCommunities(const std::unordered_map<int, std::set<int>>& communities, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    file << "# Community_ID Size Nodes" << std::endl;
    for (const auto& pair : communities) {
        file << pair.first << " " << pair.second.size() << " ";
        for (int node : pair.second) {
            file << node << " ";
        }
        file << std::endl;
    }
    
    file.close();
    std::cout << "Community structure saved to: " << filename << std::endl;
}

Graph loadDBLPGraph(){
    Graph graph;
    std::string filename = "datasets/DBLP/com-dblp.ungraph.txt";
    
    graph.loadFromFile(filename);
    return graph;

}

Graph loadFootballGraph(){
    Graph graph;
    std::string filename = "datasets/football/football.gml";
    
    graph.loadFromFileGML(filename);
    return graph;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments (if any)
    int num_ants = 20;
    int num_iterations = 50;
    int num_threads = 4;
    bool use_parallel = false;
    
    if (argc > 1) num_ants = std::stoi(argv[1]);
    if (argc > 2) num_iterations = std::stoi(argv[2]);
    if (argc > 3) num_threads = std::stoi(argv[3]);
    if (argc > 4) use_parallel = std::stoi(argv[4]) != 0;
    
    // Load graph
    std::cout << "Loading graph..." << std::endl;
    Graph graph = loadFootballGraph();
    
    std::cout << "\nRunning Edge-based ACO Community Detection with:" << std::endl;
    std::cout << "  - " << graph.getNumNodes() << " nodes" << std::endl;
    std::cout << "  - " << graph.getNumEdges() << " edges" << std::endl;
    std::cout << "  - " << num_ants << " ants" << std::endl;
    std::cout << "  - " << num_iterations << " iterations" << std::endl;
    if (use_parallel) {
        std::cout << "  - " << num_threads << " threads (parallel execution)" << std::endl;
    } else {
        std::cout << "  - Sequential execution" << std::endl;
    }
    
    // Create ACO algorithm
    ACOCommunityDetection aco(graph, num_ants, num_iterations);
    
    // Get start time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run ACO
    std::unordered_map<int, std::set<int>> communities;
    if (use_parallel) {
        // communities = aco.runParallel(num_threads);
    } else {
        communities = aco.run();
    }
    
    // Get end time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "\nACO Community Detection completed in " 
              << elapsed.count() << " seconds" << std::endl;
    
    // Print communities
    printCommunities(communities);
    
    // Save results
    std::string output_file = use_parallel ? "parallel_communities.txt" : "sequential_communities.txt";
    saveCommunities(communities, output_file);
    
    return 0;
}