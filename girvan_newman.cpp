#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <chrono>
#include <iomanip>
#include <omp.h>


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

    void saveGraph(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }
        

        file << "Graph adjacency list:" << std::endl;
        for (const auto& pair : adjList) {
            file << pair.first << " -> ";
            for (int neighbor : pair.second) {
                file << neighbor << " ";
            }
            file << std::endl;
        }

        file.close();
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

// ACO parameters structure
struct ACOParameters {
    double initialPheromone;
    double evaporationRate;
    double alpha;  // Pheromone importance
    double beta;   // Heuristic importance
    int numAnts;
    int maxIterations;
    
    ACOParameters() 
        : initialPheromone(1.0), evaporationRate(0.1), 
          alpha(1.0), beta(2.0), numAnts(10), maxIterations(5) {}
};

// Class for implementing ACO-based Girvan-Newman algorithm
class ACOGirvanNewman {
private:
    Graph graph;
    ACOParameters params;
    std::unordered_map<Edge, double> pheromones;
    std::unordered_map<Edge, double> betweenness;
    std::unordered_map<Edge, bool> removedEdges;
    std::random_device rd;

    std::mt19937& getLocalRNG() {
        static thread_local std::random_device rd;
        static thread_local std::mt19937 rng(rd());
        return rng;
    }

    // Initialize pheromones on all edges
    void initializePheromones() {
        const auto& adjList = graph.getAdjList();
        for (const auto& [node, neighbors] : adjList) {
            for (int neighbor : neighbors) {
                Edge edge(node, neighbor);
                pheromones[edge] = params.initialPheromone;
                betweenness[edge] = 0.0;
                removedEdges[edge] = false;
            }
        }
    }
    
    // Construct a path from start to end using ACO principles
    std::vector<int> constructAntPath(int start, int end) {

        std::vector<int> path;
        std::unordered_set<int> visited;
        std::mt19937& rng = getLocalRNG();

        int current = start;
        path.push_back(current);
        visited.insert(current);
        
        while (current != end) {
            const auto& neighbors = graph.getAdjList().at(current);
            std::vector<int> validNeighbors;
            std::vector<double> probabilities;
            double totalProb = 0.0;
            
            // Calculate probabilities for each unvisited neighbor
            for (int neighbor : neighbors) {
                Edge edge(current, neighbor);
                
                // Skip removed edges
                if (removedEdges[edge]) continue;
                
                // Skip visited nodes
                if (visited.find(neighbor) != visited.end()) continue;
                
                double pheromone = pheromones[edge];
                // Simple heuristic: inverse of degree (prefer less connected nodes)
                double heuristic = 1.0 / graph.getAdjList().at(neighbor).size();
                
                double prob = std::pow(pheromone, params.alpha) * 
                              std::pow(heuristic, params.beta);
                
                validNeighbors.push_back(neighbor);
                probabilities.push_back(prob);
                totalProb += prob;
            }
            
            // Dead end - backtrack one step if possible
            if (validNeighbors.empty()) {
                if (path.size() <= 1) return {}; // Can't backtrack
                
                path.pop_back();
                // visited.erase(current);
                current = path.back();
                continue;
            }
            
            // Select next node based on probabilities
            std::uniform_real_distribution<double> dist(0, totalProb);
            double r = dist(rng);
            double cumProb = 0.0;
            int nextNode = validNeighbors[0]; // Default
            
            for (size_t i = 0; i < validNeighbors.size(); ++i) {
                cumProb += probabilities[i];
                if (r <= cumProb) {
                    nextNode = validNeighbors[i];
                    break;
                }
            }
            
            current = nextNode;
            path.push_back(current);
            visited.insert(current);
            
            // Success if we reached the target
            if (current == end) break;
        }
        
        return path;
    }
    
    // Update pheromones based on ant paths
    void updatePheromones(const std::vector<std::vector<int>>& paths) {
        // Evaporation phase
        for (auto& [edge, pheromone] : pheromones) {
            pheromone *= (1.0 - params.evaporationRate);
        }
        
        // Deposition phase
        for (const auto& path : paths) {
            if (path.size() < 2) continue;
            
            // Calculate amount to deposit (inverse of path length)
            double deposit = 1.0 / (path.size() - 1);
            
            // Add pheromone to all edges in the path
            for (size_t i = 0; i < path.size() - 1; ++i) {
                Edge edge(path[i], path[i+1]);
                pheromones[edge] += deposit;
            }
        }
    }
    
    // Calculate edge betweenness using ACO
void calculateBetweenness() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Reset betweenness values
    for (auto& [edge, value] : betweenness) {
        value = 0.0;
    }
    
    std::vector<int> nodes = graph.getNodeIDs();
    
    // Create all node pairs
    std::vector<std::pair<int, int>> nodePairs;
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (size_t j = i + 1; j < nodes.size(); ++j) {
            nodePairs.push_back({nodes[i], nodes[j]});
        }
    }
    std::mt19937 shuffleRng(rd());

    // Shuffle and limit pairs if needed
    std::shuffle(nodePairs.begin(), nodePairs.end(), shuffleRng);
    int maxPairs = std::min(static_cast<int>(nodePairs.size()), 2000);
    if (nodePairs.size() > maxPairs) {
        nodePairs.resize(maxPairs);
    }
    
    // Create thread-local storage for betweenness contributions
    std::vector<std::unordered_map<Edge, double>> threadBetweenness(omp_get_max_threads());
    
    // FIRST LEVEL OF PARALLELISM: Across node pairs
    #pragma omp parallel for schedule(dynamic)
    for (size_t pairIdx = 0; pairIdx < nodePairs.size(); ++pairIdx) {
        int threadId = omp_get_thread_num();
        auto [start, end] = nodePairs[pairIdx];
        
        // Thread-local RNG to avoid contention
        thread_local std::mt19937 localRng(rd() + omp_get_thread_num() * 1000);
        
        for (int iter = 0; iter < params.maxIterations; ++iter) {
            std::vector<std::vector<int>> iterPaths;
            
            // Pre-allocate to avoid race conditions
            std::vector<std::vector<int>> antPaths(params.numAnts);
            
            // SECOND LEVEL OF PARALLELISM: Across ants
            #pragma omp parallel for schedule(dynamic) num_threads(4)
            for (int ant = 0; ant < params.numAnts; ++ant) {
                // Each ant needs its own RNG
                thread_local std::mt19937 antRng(rd() + omp_get_thread_num() * 100 + ant);
                
                // Construct path using local RNG
                std::vector<int> path;
                
                // Modified version of constructAntPath that uses the provided RNG
                // This is essentially inline code that would normally be in constructAntPath
                {
                    path.clear();
                    std::unordered_set<int> visited;
                    
                    int current = start;
                    path.push_back(current);
                    visited.insert(current);
                    
                    while (current != end) {
                        const auto& neighbors = graph.getAdjList().at(current);
                        std::vector<int> validNeighbors;
                        std::vector<double> probabilities;
                        double totalProb = 0.0;
                        
                        // Calculate probabilities for each unvisited neighbor
                        for (int neighbor : neighbors) {
                            Edge edge(current, neighbor);
                            
                            // Skip removed edges
                            if (removedEdges[edge]) continue;
                            
                            // Skip visited nodes
                            if (visited.find(neighbor) != visited.end()) continue;
                            
                            double pheromone = pheromones[edge];
                            // Simple heuristic: inverse of degree
                            double heuristic = 1.0 / graph.getAdjList().at(neighbor).size();
                            
                            double prob = std::pow(pheromone, params.alpha) * 
                                          std::pow(heuristic, params.beta);
                            
                            validNeighbors.push_back(neighbor);
                            probabilities.push_back(prob);
                            totalProb += prob;
                        }
                        
                        // Dead end - backtrack one step if possible
                        if (validNeighbors.empty()) {
                            if (path.size() <= 1) {
                                path.clear();
                                break; // Can't backtrack
                            }
                            
                            path.pop_back();
                            current = path.back();
                            continue;
                        }
                        
                        // Select next node based on probabilities
                        std::uniform_real_distribution<double> dist(0, totalProb);
                        double r = dist(antRng);
                        double cumProb = 0.0;
                        int nextNode = validNeighbors[0]; // Default
                        
                        for (size_t i = 0; i < validNeighbors.size(); ++i) {
                            cumProb += probabilities[i];
                            if (r <= cumProb) {
                                nextNode = validNeighbors[i];
                                break;
                            }
                        }
                        
                        current = nextNode;
                        path.push_back(current);
                        visited.insert(current);
                        
                        // Success if we reached the target
                        if (current == end) break;
                    }
                }
                
                if (!path.empty()) {
                    antPaths[ant] = path;
                }
            }
            
            // Collect valid paths
            for (const auto& path : antPaths) {
                if (!path.empty()) {
                    iterPaths.push_back(path);
                }
            }
            
            // Update pheromones - this needs synchronization
            #pragma omp critical
            {
                updatePheromones(iterPaths);
            }
            
            // Process paths for this iteration
            for (const auto& path : iterPaths) {
                if (path.size() < 2) continue;
                
                // Each edge in the path gets a betweenness contribution
                for (size_t i = 0; i < path.size() - 1; ++i) {
                    Edge edge(path[i], path[i+1]);
                    threadBetweenness[threadId][edge] += 1.0 / iterPaths.size();
                }
            }
        }
        
        // Print progress periodically
        if (pairIdx % 100 == 0) {
            #pragma omp critical
            {
                std::cout << "Processed " << pairIdx << " of " << nodePairs.size() << " node pairs" << std::endl;
            }
        }
    }
    
    // Combine thread-local betweenness values
    for (const auto& localMap : threadBetweenness) {
        for (const auto& [edge, value] : localMap) {
            betweenness[edge] += value;
        }
    }
    
    // Normalize betweenness
    double maxBetweenness = 0.0;
    for (const auto& [edge, value] : betweenness) {
        maxBetweenness = std::max(maxBetweenness, value);
    }
    
    if (maxBetweenness > 0) {
        for (auto& [edge, value] : betweenness) {
            value /= maxBetweenness;
        }
    }
    
    // Print timing results
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "calculateBetweenness took " << duration.count() << " seconds" << std::endl;
}

    // Find edge with highest betweenness
    Edge findHighestBetweennessEdge() {
        Edge maxEdge(-1, -1);
        double maxValue = -1.0;
        
        for (const auto& [edge, value] : betweenness) {
            if (!removedEdges[edge] && value > maxValue) {
                maxValue = value;
                maxEdge = edge;
            }
        }
        
        return maxEdge;
    }
    
    // Mark an edge as removed
    void removeEdge(const Edge& edge) {
        removedEdges[edge] = true;
    }
    
    // Find communities (connected components after edge removals)
std::unordered_map<int, std::set<int>> findConnectedComponents() {
    std::unordered_map<int, std::set<int>> components;
    std::vector<int> nodes = graph.getNodeIDs();
    std::unordered_set<int> visited;
    int communityId = 0;
    
    for (int node : nodes) {
        if (visited.find(node) == visited.end()) {
            // Start a new component
            std::set<int> component;
            std::queue<int> queue;
            
            queue.push(node);
            visited.insert(node);
            
            while (!queue.empty()) {
                int current = queue.front();
                queue.pop();
                component.insert(current);
                
                // Add unvisited neighbors via non-removed edges
                const auto& neighbors = graph.getAdjList().at(current);
                for (int neighbor : neighbors) {
                    Edge edge(current, neighbor);
                    if (!removedEdges[edge] && visited.find(neighbor) == visited.end()) {
                        queue.push(neighbor);
                        visited.insert(neighbor);
                    }
                }
            }
            
            components[communityId] = component;
            communityId++;
        }
    }
    
    return components;
}


    // Calculate modularity of a partitioning
double calculateModularity(const std::unordered_map<int, std::set<int>>& communities) {
    double modularity = 0.0;
    int totalEdges = graph.getNumEdges();
    
    for (const auto& [communityId, community] : communities) {
        int edgesWithin = 0;
        int totalDegree = 0;
        
        // Calculate edges within community and total degree
        for (int node : community) {
            const auto& neighbors = graph.getAdjList().at(node);
            totalDegree += neighbors.size();
            
            for (int neighbor : neighbors) {
                Edge edge(node, neighbor);
                if (!removedEdges[edge] && 
                    community.find(neighbor) != community.end()) {
                    edgesWithin++;
                }
            }
        }
        
        // Each edge is counted twice in undirected graph
        edgesWithin /= 2;
        
        double edgeFraction = (double)edgesWithin / totalEdges;
        double degreeFraction = (double)totalDegree / (2 * totalEdges);
        
        modularity += edgeFraction - (degreeFraction * degreeFraction);
    }
    
    return modularity;
}

public:
    ACOGirvanNewman(const Graph& g, const ACOParameters& p) 
        : graph(g), params(p) /*, rng(rd()) */{
        initializePheromones();
    }


    
    
    // Main algorithm to detect communities
std::unordered_map<int, std::set<int>> detectCommunities() {
    std::unordered_map<int, std::set<int>> bestCommunities;
    double bestModularity = -1.0;
    int edgesRemoved = 0;
    int maxEdgesToRemove = graph.getNumEdges() / 2; // Limit for efficiency
    
    std::cout << "Starting community detection..." << std::endl;
    
    while (edgesRemoved < maxEdgesToRemove) {
        // Calculate betweenness using ACO
        calculateBetweenness();
        
        // Find edge with highest betweenness
        Edge maxEdge = findHighestBetweennessEdge();
        if (maxEdge.from == -1 || maxEdge.to == -1) {
            break; // No more edges to remove
        }
        
        std::cout << "Removing edge: " << maxEdge.from << " - " << maxEdge.to << std::endl;
        
        // Remove the edge
        removeEdge(maxEdge);
        edgesRemoved++;
        
        // Find current communities
        std::unordered_map<int, std::set<int>> currentCommunities = findConnectedComponents();
        
        std::cout << "Communities found: " << std::endl;
        for (const auto& [communityId, nodes] : currentCommunities) {
            std::cout << "Community " << communityId << ": ";
            for (int node : nodes) {
                std::cout << node << " ";
            }
            std::cout << std::endl;
        }
        
        // Calculate modularity
        double currentModularity = calculateModularity(currentCommunities);
        
        std::cout << "Edge removals: " << edgesRemoved 
                 << ", Communities: " << currentCommunities.size() 
                 << ", Modularity: " << currentModularity << std::endl;
        
        // Update best solution if modularity improved
        if (currentModularity > bestModularity) {
            bestModularity = currentModularity;
            bestCommunities = currentCommunities;
            
            std::cout << "New best solution found: " << bestCommunities.size() 
                     << " communities, modularity = " << bestModularity << std::endl;
        }

        // if (edgesRemoved == 35) {
        //     return bestCommunities;
        // } 
    }
    
    std::cout << "Community detection completed." << std::endl;
    std::cout << "Best partition: " << bestCommunities.size() 
             << " communities with modularity " << bestModularity << std::endl;
    
    return bestCommunities;
}
};

void saveGraphWithCommunities(const Graph& graph, 
                             const std::unordered_map<int, std::set<int>>& communities, 
                             const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // Create a mapping from node to community
    std::unordered_map<int, int> node_to_community;
    for (const auto& [communityId, nodes] : communities) {
        for (int node : nodes) {
            node_to_community[node] = communityId;
        }
    }
    
    // Save adjacency list with community assignments
    const auto& adj_list = graph.getAdjList();
    
    file << "Graph adjacency list:" << std::endl;
    for (const auto& pair : adj_list) {
        int node = pair.first;
        file << node << " -> ";
        for (int neighbor : pair.second) {
            file << neighbor << " ";
        }
        file << std::endl;
    }
    
    // Save community assignments
    file << "# Community_ID Size Nodes" << std::endl;
    for (const auto& [communityId, nodes] : communities) {
        file << communityId << " " << nodes.size() << " ";
        for (int node : nodes) {
            file << node << " ";
        }
        file << std::endl;
    }
    
    file.close();
    std::cout << "Graph with communities saved to: " << filename << std::endl;
}

Graph loadTestGraph(){
    Graph graph;
    std::string filename = "datasets/test/test_graph.gml";
    
    graph.loadFromFileGML(filename);
    return graph;
}

Graph loadFootballGraph(){
    Graph graph;
    std::string filename = "datasets/football/football.gml";
    
    graph.loadFromFileGML(filename);
    return graph;
}

int main(int argc, char* argv[]) {
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
    // Graph graph = loadFootballGraph();//loadTestGraph();
     Graph graph = loadTestGraph();
    
    // Set up ACO parameters
    ACOParameters params;
    params.initialPheromone = 1.0;
    params.evaporationRate = 0.2;
    params.alpha = 1.0;
    params.beta = 2.0;
    params.numAnts = num_ants;
    params.maxIterations = num_iterations;
    
    std::cout << "Graph loaded: " << graph.getNumNodes() << " nodes, " 
              << graph.getNumEdges() << " edges" << std::endl;
    std::cout << "ACO parameters: alpha=" << params.alpha << ", beta=" << params.beta 
              << ", evapRate=" << params.evaporationRate 
              << ", ants=" << params.numAnts
              << ", iterations=" << params.maxIterations << std::endl;
    
    // Initialize and run community detection
    ACOGirvanNewman detector(graph, params);
    std::unordered_map<int, std::set<int> > communities = detector.detectCommunities();
    
    // // Print results
    // std::cout << "\nDetected " << communities.size() << " communities:" << std::endl;
    // for (size_t i = 0; i < communities.size(); i++) {
    //     std::cout << "Community " << i + 1 << " (size " << communities[i].size() << "): ";
        
    //     // Print up to 10 nodes per community to avoid overwhelming output
    //     for (size_t j = 0; j < std::min(communities[i].size(), size_t(10)); j++) {
    //         std::cout << communities[i][j] << " ";
    //     }
        
    //     if (communities[i].size() > 10) {
    //         std::cout << "... (+" << communities[i].size() - 10 << " more nodes)";
    //     }
        
    //     std::cout << std::endl;
    // }
    
    saveGraphWithCommunities(graph, communities, "sequential_communities.txt");
    return 0;
}