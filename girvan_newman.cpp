#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <chrono>

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
    std::mt19937 rng;
    
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
        // Reset betweenness values
        for (auto& [edge, value] : betweenness) {
            value = 0.0;
        }
        
        std::vector<int> nodes = graph.getNodeIDs();
        int totalPairs = 0;
        
        // Sample node pairs for efficiency in large graphs
        int maxPairs = std::min(static_cast<int>(nodes.size() * (nodes.size() - 1) / 2), 
                               2000); // Limit pairs sampled
        
        std::vector<std::pair<int, int>> nodePairs;
        for (size_t i = 0; i < nodes.size(); ++i) {
            for (size_t j = i + 1; j < nodes.size(); ++j) {
                nodePairs.push_back({nodes[i], nodes[j]});
            }
        }


        std::shuffle(nodePairs.begin(), nodePairs.end(), rng);
        if (nodePairs.size() > maxPairs) {
            nodePairs.resize(maxPairs);
        }
        
        // std::cout << "Node pairs after shuffling: \n";

        // for(auto pair: nodePairs) {
        //     std::cout << pair.first <<", "<< pair.second << "\n";
        // }

        // Process each node pair
        for (const auto& [start, end] : nodePairs) {
            std::vector<std::vector<int>> allPaths;
            
            // Run ACO iterations
            for (int iter = 0; iter < params.maxIterations; ++iter) {
                std::vector<std::vector<int>> iterPaths;
                
                // Deploy multiple ants
                for (int ant = 0; ant < params.numAnts; ++ant) {
                    std::cout << "Constructing path for : "<< start << "->" <<end <<"\n";
                    std::vector<int> path = constructAntPath(start, end);
                    for(auto node :path) {
                        std::cout << node << " \n";
                    }
                    if (!path.empty()) {
                        iterPaths.push_back(path);
                    }
                }
                
                // Update pheromones based on paths found
                updatePheromones(iterPaths);
                
                // Collect paths for betweenness calculation
                allPaths.insert(allPaths.end(), iterPaths.begin(), iterPaths.end());
            }
            
            totalPairs += !allPaths.empty();
            
            // Update edge betweenness based on paths
            for (const auto& path : allPaths) {
                if (path.size() < 2) continue;
                
                // Each edge in the path gets a betweenness contribution
                for (size_t i = 0; i < path.size() - 1; ++i) {
                    Edge edge(path[i], path[i+1]);
                    betweenness[edge] += 1.0 / allPaths.size();
                }
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

        std::cout <<"Normalized betweennness : \n";
        for (const auto& [edge, value] : betweenness) {
            std::cout<<"EDGE " << edge.from << " -> " << edge.to << " : " << value << "\n";
        }

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
    std::vector<std::vector<int>> findConnectedComponents() {
        std::vector<std::vector<int>> components;
        std::vector<int> nodes = graph.getNodeIDs();
        std::unordered_set<int> visited;
        
        for (int node : nodes) {
            if (visited.find(node) == visited.end()) {
                // Start a new component
                std::vector<int> component;
                std::queue<int> queue;
                
                queue.push(node);
                visited.insert(node);
                
                while (!queue.empty()) {
                    int current = queue.front();
                    queue.pop();
                    component.push_back(current);
                    
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
                
                components.push_back(component);
            }
        }
        
        return components;
    }
    
    // Calculate modularity of a partitioning
    double calculateModularity(const std::vector<std::vector<int>>& communities) {
        double modularity = 0.0;
        int totalEdges = graph.getNumEdges();
        
        for (const auto& community : communities) {
            int edgesWithin = 0;
            int totalDegree = 0;
            
            // Calculate edges within community and total degree
            for (int node : community) {
                const auto& neighbors = graph.getAdjList().at(node);
                totalDegree += neighbors.size();
                
                for (int neighbor : neighbors) {
                    Edge edge(node, neighbor);
                    if (!removedEdges[edge] && 
                        std::find(community.begin(), community.end(), neighbor) != community.end()) {
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
        : graph(g), params(p), rng(rd()) {
        initializePheromones();
    }
    
    // Main algorithm to detect communities
    std::vector<std::vector<int>> detectCommunities() {
        std::vector<std::vector<int>> bestCommunities;
        double bestModularity = -1.0;
        int edgesRemoved = 0;
        int maxEdgesToRemove = graph.getNumEdges() / 2; // Limit for efficiency
        int iterCount = 0;
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
            std::vector<std::vector<int>> currentCommunities = findConnectedComponents();
            std::cout<<"Communities found: \n";
            for(auto comm : currentCommunities) {
                for(auto node : comm) {
                    std :: cout << node << " " ;
                }
                std::cout<<"\n";
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
            exit(0);
        }
        
        std::cout << "Community detection completed." << std::endl;
        std::cout << "Best partition: " << bestCommunities.size() 
                 << " communities with modularity " << bestModularity << std::endl;
        
        return bestCommunities;
    }
};


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
    Graph graph = loadTestGraph();
    //loadTestGraph();
    
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
    std::vector<std::vector<int>> communities = detector.detectCommunities();
    
    // Print results
    std::cout << "\nDetected " << communities.size() << " communities:" << std::endl;
    for (size_t i = 0; i < communities.size(); i++) {
        std::cout << "Community " << i + 1 << " (size " << communities[i].size() << "): ";
        
        // Print up to 10 nodes per community to avoid overwhelming output
        for (size_t j = 0; j < std::min(communities[i].size(), size_t(10)); j++) {
            std::cout << communities[i][j] << " ";
        }
        
        if (communities[i].size() > 10) {
            std::cout << "... (+" << communities[i].size() - 10 << " more nodes)";
        }
        
        std::cout << std::endl;
    }
    
    
    return 0;
}