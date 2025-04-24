#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <random>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <mpi.h>

// Graph class (similar to original, with serialization support for MPI)
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
    
void loadSoftwareFromFileGML(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    std::map<int, bool> nodes; // To keep track of nodes we've seen
    int edgeCount = 0;
    std::string line;
    bool inGraph = false;
    bool inNode = false;
    bool inEdge = false;
    int currentNodeId = -1;
    int sourceNode = -1;
    int targetNode = -1;
    std::string token;
    while (std::getline(file, line)) {
        // Trim leading and trailing whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        if (line.empty()) continue; // Skip empty lines
        std::istringstream iss(line);
        iss >> token;
        // Parse GML format
        if (token == "graph") {
            inGraph = true;
        } else if (token == "directed") {
            // Handle directed graph property
            int directed;
            iss >> directed;
            // You can store this information if needed
        } else if (token == "node") {
            inNode = true;
            inEdge = false;
        } else if (token == "edge") {
            inNode = false;
            inEdge = true;
        } else if (token == "[") {
            // Opening bracket, nothing specific to do
            continue;
        } else if (token == "]") {
            // Closing bracket
            if (inNode) {
                // Just mark that we've left the node section
                inNode = false;
            } else if (inEdge) {
                // Finish processing the current edge
                if (sourceNode != -1 && targetNode != -1) {
                    addEdge(sourceNode, targetNode);
                    edgeCount++;
                    sourceNode = -1;
                    targetNode = -1;
                }
                inEdge = false;
            } else if (inGraph) {
                // End of graph
                inGraph = false;
            }
        } else if (inNode && token == "id") {
            // Parse node ID - assumes ID is on the same line
            iss >> currentNodeId;
            // Mark that we've seen this node
            nodes[currentNodeId] = true;
        } else if (inEdge && token == "source") {
            // Parse edge source - assumes source is on the same line
            iss >> sourceNode;
        } else if (inEdge && token == "target") {
            // Parse edge target - assumes target is on the same line
            iss >> targetNode;
        }
        // Ignore other properties like _pos for now
    }
    file.close();
    std::cout << "Graph loaded successfully from GML!" << std::endl;
    std::cout << "Nodes: " << nodes.size() << ", Edges: " << edgeCount << std::endl;
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
    }    int getNumNodes() const { return numNodes; }
    int getNumEdges() const { return numEdges / 2; }
    
    const std::unordered_map<int, std::vector<int>>& getAdjList() const {
        return adjList;
    }
    
    std::vector<int> getNodeIDs() const {
        std::vector<int> nodes;
        for (const auto& pair : adjList) {
            nodes.push_back(pair.first);
        }
        return nodes;
    }
    
    // Serialize graph for MPI communication
    void serializeGraph(std::vector<int>& buffer) {
        buffer.clear();
        
        // Add number of nodes and edges
        buffer.push_back(numNodes);
        buffer.push_back(numEdges);
        
        // Add adjacency list
        for (const auto& pair : adjList) {
            int node = pair.first;
            const auto& neighbors = pair.second;
            
            buffer.push_back(node);
            buffer.push_back(neighbors.size());
            
            for (int neighbor : neighbors) {
                buffer.push_back(neighbor);
            }
        }
    }
    
    // Deserialize graph from MPI buffer
    void deserializeGraph(const std::vector<int>& buffer) {
        adjList.clear();
        
        size_t pos = 0;
        numNodes = buffer[pos++];
        numEdges = buffer[pos++];
        
        while (pos < buffer.size()) {
            int node = buffer[pos++];
            int numNeighbors = buffer[pos++];
            
            for (int i = 0; i < numNeighbors; i++) {
                int neighbor = buffer[pos++];
                adjList[node].push_back(neighbor);
            }
        }
    }
};

// Structure to represent a solution (community assignment)
struct Solution {
    std::unordered_map<int, int> community; // Map: node_id -> community_id
    double modularity;
    
    Solution() : modularity(0.0) {}
    
    explicit Solution(const std::vector<int>& nodes) : modularity(0.0) {
        for (int node : nodes) {
            community[node] = node; // Initially node ID is community ID
        }
    }
    
    // Serialize solution for MPI communication
    void serializeSolution(std::vector<int>& intBuffer, std::vector<double>& doubleBuffer) {
        intBuffer.clear();
        doubleBuffer.clear();
        
        // Add modularity to double buffer
        doubleBuffer.push_back(modularity);
        
        // Add community assignments to int buffer
        for (const auto& pair : community) {
            intBuffer.push_back(pair.first);   // node ID
            intBuffer.push_back(pair.second);  // community ID
        }
    }
    
    // Deserialize solution from MPI buffers
    void deserializeSolution(const std::vector<int>& intBuffer, const std::vector<double>& doubleBuffer) {
        community.clear();
        
        // Get modularity
        modularity = doubleBuffer[0];
        
        // Get community assignments
        for (size_t i = 0; i < intBuffer.size(); i += 2) {
            int nodeID = intBuffer[i];
            int communityID = intBuffer[i + 1];
            community[nodeID] = communityID;
        }
    }
};

// Random number generator with seed for consistent behavior across processes
class RandomGenerator {
private:
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;

public:
    RandomGenerator() : rng(std::random_device()()), dist(0.0, 1.0) {}
    
    explicit RandomGenerator(unsigned int seed) : rng(seed), dist(0.0, 1.0) {}
    
    double random() {
        return dist(rng);
    }
    
    template<typename T>
    void shuffle(std::vector<T>& v) {
        std::shuffle(v.begin(), v.end(), rng);
    }
};

class MPINodeCommunityACO {
private:
    Graph graph;
    int num_ants;
    int max_iterations;
    double alpha;     // Pheromone influence
    double beta;      // Heuristic influence
    double rho;       // Evaporation rate
    double q0;        // Probability of exploitation vs exploration
    int sync_frequency; // How often to synchronize pheromones
    
    // MPI variables
    int rank;
    int size;
    int ants_per_process;
    
    // Pheromone matrix: pheromone[node][community] = pheromone level
    std::unordered_map<int, std::unordered_map<int, double>> pheromones;
    
    // Best solution found so far locally
    Solution best_local_solution;
    
    // Best solution found globally
    Solution best_global_solution;
    
    // Random number generator
    RandomGenerator rng;
    
    // Initialize pheromone values
    void initializePheromones() {
        std::vector<int> nodes = graph.getNodeIDs();
        double initial_pheromone = 1.0 / nodes.size();
        
        if (rank == 0) {
            std::cout << "Initializing pheromones..." << std::endl;
        }
        
        for (int node : nodes) {
            for (int possible_community : nodes) {
                pheromones[node][possible_community] = initial_pheromone;
            }
        }
        
        if (rank == 0) {
            std::cout << "Initialization complete" << std::endl;
        }
    }
    
    // Print statistics about the pheromone matrix (only rank 0)
    void printPheromoneStats() {
        if (rank != 0) return;
        
        double min_val = std::numeric_limits<double>::max();
        double max_val = 0.0;
        double sum = 0.0;
        int count = 0;
        
        for (const auto& node_map : pheromones) {
            for (const auto& comm_val : node_map.second) {
                min_val = std::min(min_val, comm_val.second);
                max_val = std::max(max_val, comm_val.second);
                sum += comm_val.second;
                count++;
            }
        }
        
        double avg = count > 0 ? sum / count : 0.0;
        
        std::cout << "Pheromone matrix stats:" << std::endl;
        std::cout << "  Min: " << min_val << std::endl;
        std::cout << "  Max: " << max_val << std::endl;
        std::cout << "  Avg: " << avg << std::endl;
        std::cout << "  Total entries: " << count << std::endl;
    }
    
    // Calculate how many connections node has to a given community
    int connectionsToCommunity(int node, int community_id, const Solution& solution) {
        const auto& adj_list = graph.getAdjList();
        
        if (adj_list.find(node) == adj_list.end()) {
            return 0;
        }
        
        int connections = 0;
        for (int neighbor : adj_list.at(node)) {
            if (solution.community.find(neighbor) != solution.community.end() && 
                solution.community.at(neighbor) == community_id) {
                connections++;
            }
        }
        
        return connections;
    }
    
    // Construct a solution using ant colony principles
    Solution constructSolution(int ant_id) {
        std::vector<int> nodes = graph.getNodeIDs();
        Solution solution(nodes);
        const auto& adj_list = graph.getAdjList();
        
        // Shuffle nodes to process them in random order
        rng.shuffle(nodes);
        
        // First ant uses initial solution, others construct new ones
        if (ant_id > 0) {
            // For each node, decide which community to join
            for (int node : nodes) {
                // Get potential communities (start with all neighbors' communities)
                std::set<int> potential_communities;
                potential_communities.insert(node); // Own community is always an option
                
                if (adj_list.find(node) != adj_list.end()) {
                    for (int neighbor : adj_list.at(node)) {
                        potential_communities.insert(solution.community[neighbor]);
                    }
                }
                
                // Decide whether to exploit or explore
                double q = rng.random();
                
                if (q < q0) {
                    // Exploitation: choose best community by deterministic rule
                    int best_community = node;
                    double best_value = 0.0;
                    
                    for (int comm : potential_communities) {
                        // Pheromone level
                        double pheromone = pheromones[node][comm];
                        
                        // Heuristic: number of connections to community
                        int connections = connectionsToCommunity(node, comm, solution);
                        double heuristic = std::max(0.1, static_cast<double>(connections));
                        
                        // Combined value using ACO formula
                        double value = std::pow(pheromone, alpha) * std::pow(heuristic, beta);
                        
                        if (value > best_value) {
                            best_value = value;
                            best_community = comm;
                        }
                    }
                    
                    // Assign node to the best community
                    solution.community[node] = best_community;
                }
                else {
                    // Exploration: probabilistic choice
                    std::vector<int> communities;
                    std::vector<double> probabilities;
                    double total = 0.0;
                    
                    for (int comm : potential_communities) {
                        // Pheromone level
                        double pheromone = pheromones[node][comm];
                        
                        // Heuristic: number of connections to community
                        int connections = connectionsToCommunity(node, comm, solution);
                        double heuristic = std::max(0.1, static_cast<double>(connections));
                        
                        // Combined value using ACO formula
                        double value = std::pow(pheromone, alpha) * std::pow(heuristic, beta);
                        
                        communities.push_back(comm);
                        probabilities.push_back(value);
                        total += value;
                    }
                    
                    // Normalize probabilities
                    if (total > 0) {
                        for (auto& p : probabilities) {
                            p /= total;
                        }
                    }
                    else {
                        // If all values are zero, use uniform distribution
                        for (auto& p : probabilities) {
                            p = 1.0 / probabilities.size();
                        }
                    }
                    
                    // Select community using roulette wheel selection
                    double r = rng.random();
                    double cumulative = 0.0;
                    int selected_community = communities[0];
                    
                    for (size_t i = 0; i < communities.size(); i++) {
                        cumulative += probabilities[i];
                        if (r <= cumulative) {
                            selected_community = communities[i];
                            break;
                        }
                    }
                    
                    // Assign node to the selected community
                    solution.community[node] = selected_community;
                }
            }
        }
        
        // Calculate solution modularity
        solution.modularity = calculateModularity(solution);
        
        return solution;
    }
    
    // Update pheromone levels based on solutions
    void updatePheromones(const std::vector<Solution>& solutions) {
        // Evaporation
        for (auto& node_map : pheromones) {
            for (auto& comm_pair : node_map.second) {
                comm_pair.second *= (1.0 - rho);
            }
        }
        
        // Add new pheromones based on solution quality
        for (const auto& solution : solutions) {
            double delta = solution.modularity; // Use modularity as deposit amount
            
            // For each node-community assignment in the solution
            for (const auto& pair : solution.community) {
                int node = pair.first;
                int comm = pair.second;
                
                // Add pheromone proportional to solution quality
                pheromones[node][comm] += delta;
            }
        }
        
        // Normalize pheromone values to prevent extreme differences
        double max_pheromone = 0.0;
        for (const auto& node_map : pheromones) {
            for (const auto& comm_val : node_map.second) {
                max_pheromone = std::max(max_pheromone, comm_val.second);
            }
        }
        
        if (max_pheromone > 10.0) {
            double scale_factor = 5.0 / max_pheromone;
            
            for (auto& node_map : pheromones) {
                for (auto& comm_pair : node_map.second) {
                    comm_pair.second *= scale_factor;
                }
            }
        }
    }
    
    // Calculate modularity of a solution
    double calculateModularity(const Solution& solution) {
        const auto& adj_list = graph.getAdjList();
        double m = graph.getNumEdges(); // Total number of edges
        double q = 0.0;
        
        // Calculate the sum of degrees for each community
        std::unordered_map<int, double> community_degrees;
        for (const auto& pair : adj_list) {
            int node = pair.first;
            
            if (solution.community.find(node) == solution.community.end()) {
                continue; // Skip nodes not in solution
            }
            
            int comm = solution.community.at(node);
            if (community_degrees.find(comm) == community_degrees.end()) {
                community_degrees[comm] = 0.0;
            }
            community_degrees[comm] += pair.second.size();
        }
        
        // For each edge
        for (const auto& pair : adj_list) {
            int i = pair.first;
            
            if (solution.community.find(i) == solution.community.end()) {
                continue;
            }
            
            int c_i = solution.community.at(i);
            
            for (int j : pair.second) {
                // Process each edge once (for undirected graph)
                if (i < j) {
                    if (solution.community.find(j) == solution.community.end()) {
                        continue;
                    }
                    
                    int c_j = solution.community.at(j);
                    
                    // Same community contribution
                    if (c_i == c_j) {
                        double expected = (pair.second.size() * adj_list.at(j).size()) / (2.0 * m);
                        q += 1.0 - expected;
                    }
                }
            }
        }
        
        q /= (2.0 * m);
        return q;
    }
    
    // Synchronize best solutions across all processes
    void synchronizeBestSolutions() {
        // Serialize local best solution
        std::vector<int> localIntBuffer;
        std::vector<double> localDoubleBuffer;
        best_local_solution.serializeSolution(localIntBuffer, localDoubleBuffer);
        
        // Gather size information
        int localIntSize = localIntBuffer.size();
        int localDoubleSize = localDoubleBuffer.size();
        
        std::vector<int> intSizes(size);
        std::vector<int> doubleSizes(size);
        
        // Gather buffer sizes from all processes
        MPI_Allgather(&localIntSize, 1, MPI_INT, intSizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(&localDoubleSize, 1, MPI_INT, doubleSizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        // Calculate displacements for gatherv
        std::vector<int> intDispl(size);
        std::vector<int> doubleDispl(size);
        
        int totalIntSize = 0;
        int totalDoubleSize = 0;
        
        for (int i = 0; i < size; i++) {
            intDispl[i] = totalIntSize;
            doubleDispl[i] = totalDoubleSize;
            totalIntSize += intSizes[i];
            totalDoubleSize += doubleSizes[i];
        }
        
        // Allocate buffers for all solutions
        std::vector<int> allIntBuffer(totalIntSize);
        std::vector<double> allDoubleBuffer(totalDoubleSize);
        
        // Gather all serialized solutions
        MPI_Allgatherv(localIntBuffer.data(), localIntSize, MPI_INT,
                      allIntBuffer.data(), intSizes.data(), intDispl.data(), MPI_INT,
                      MPI_COMM_WORLD);
        
        MPI_Allgatherv(localDoubleBuffer.data(), localDoubleSize, MPI_DOUBLE,
                      allDoubleBuffer.data(), doubleSizes.data(), doubleDispl.data(), MPI_DOUBLE,
                      MPI_COMM_WORLD);
        
        // Find the best solution among all processes
        double bestModularity = best_local_solution.modularity;
        int bestRank = rank;
        
        for (int i = 0; i < size; i++) {
            if (i == rank) continue;
            
            // Get the modularity value for this process's solution
            double otherModularity = allDoubleBuffer[doubleDispl[i]];
            
            if (otherModularity > bestModularity) {
                bestModularity = otherModularity;
                bestRank = i;
            }
        }
        
        // If a better solution was found, extract it
        if (bestRank != rank) {
            // Extract the best solution from the gathered buffers
            std::vector<int> bestIntBuffer(intSizes[bestRank]);
            std::vector<double> bestDoubleBuffer(doubleSizes[bestRank]);
            
            std::copy(allIntBuffer.begin() + intDispl[bestRank],
                     allIntBuffer.begin() + intDispl[bestRank] + intSizes[bestRank],
                     bestIntBuffer.begin());
            
            std::copy(allDoubleBuffer.begin() + doubleDispl[bestRank],
                     allDoubleBuffer.begin() + doubleDispl[bestRank] + doubleSizes[bestRank],
                     bestDoubleBuffer.begin());
            
            // Update best global solution
            best_global_solution.deserializeSolution(bestIntBuffer, bestDoubleBuffer);
            
            // Update local pheromones based on the global best solution
            updatePheromones({best_global_solution});
        } else {
            best_global_solution = best_local_solution;
        }
    }
    
    // Convert solution to community map format
    std::unordered_map<int, std::set<int>> convertToCommunityMap(const Solution& solution) {
        std::unordered_map<int, std::set<int>> result;
        
        for (const auto& pair : solution.community) {
            int node = pair.first;
            int comm = pair.second;
            
            if (result.find(comm) == result.end()) {
                result[comm] = std::set<int>();
            }
            
            result[comm].insert(node);
        }
        
        return result;
    }

public:
    MPINodeCommunityACO(int r, int s, const Graph& g, int ants = 20, int iterations = 100, 
                        double a = 1.0, double b = 2.0, double r_val = 0.1, double q = 0.9,
                        int sync_freq = 5)
        : rank(r), size(s), graph(g), num_ants(ants), max_iterations(iterations),
          alpha(a), beta(b), rho(r_val), q0(q), sync_frequency(sync_freq) {
        
        // Set up random number generator with different seeds per process
        rng = RandomGenerator(std::random_device{}() + rank);
        
        // Calculate how many ants per process
        ants_per_process = num_ants / size;
        if (rank < num_ants % size) {
            ants_per_process++;
        }
        
        initializePheromones();
        
        // Initialize best solutions
        std::vector<int> nodes = graph.getNodeIDs();
        best_local_solution = Solution(nodes);
        best_local_solution.modularity = calculateModularity(best_local_solution);
        best_global_solution = best_local_solution;
    }
    
    std::unordered_map<int, std::set<int>> run() {
        if (rank == 0) {
            std::cout << "Running MPI Node-Community ACO with:" << std::endl;
            std::cout << "  - " << num_ants << " ants (" << ants_per_process << " per process)" << std::endl;
            std::cout << "  - " << max_iterations << " iterations" << std::endl;
            std::cout << "  - " << size << " MPI processes" << std::endl;
            std::cout << "  - Alpha (pheromone influence): " << alpha << std::endl;
            std::cout << "  - Beta (heuristic influence): " << beta << std::endl;
            std::cout << "  - Rho (evaporation rate): " << rho << std::endl;
            std::cout << "  - q0 (exploitation probability): " << q0 << std::endl;
            std::cout << "  - Sync frequency: " << sync_frequency << std::endl;
        }
        
        for (int iter = 0; iter < max_iterations; iter++) {
            // Each process constructs solutions with its ants
            std::vector<Solution> ant_solutions(ants_per_process);
            
            for (int ant = 0; ant < ants_per_process; ant++) {
                ant_solutions[ant] = constructSolution(ant);
                
                // Update local best solution if improved
                if (ant_solutions[ant].modularity > best_local_solution.modularity) {
                    best_local_solution = ant_solutions[ant];
                }
            }
            
            // Update local pheromones based on solutions
            updatePheromones(ant_solutions);
            
            // Periodically synchronize best solutions
            if (iter % sync_frequency == 0 || iter == max_iterations - 1) {
                synchronizeBestSolutions();
                
                if (rank == 0 && (iter % 10 == 0 || iter == max_iterations - 1)) {
                    auto communities = convertToCommunityMap(best_global_solution);
                    std::cout << "Iteration " << iter << ": " 
                              << communities.size() << " communities, "
                              << "modularity = " << best_global_solution.modularity << std::endl;
                    
                    if (iter % 10 == 0) {
                        printPheromoneStats();
                    }
                }
            }
        }
        
        // Ensure all processes have the final best solution
        synchronizeBestSolutions();
        
        // Only rank 0 reports final results
        if (rank == 0) {
            auto communities = convertToCommunityMap(best_global_solution);
            std::cout << "Final result: " << communities.size() 
                      << " communities, modularity = " << best_global_solution.modularity << std::endl;
            return communities;
        } else {
            return std::unordered_map<int, std::set<int>>();
        }
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
    
    // Print all communities, or top 10 if there are many
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

void saveGraphWithCommunities(const Graph& graph, const std::unordered_map<int, std::set<int>>& communities, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // Create a mapping from node to community
    std::unordered_map<int, int> node_to_community;
    for (const auto& pair : communities) {
        int community_id = pair.first;
        for (int node : pair.second) {
            node_to_community[node] = community_id;
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
    for (const auto& pair : communities) {
        int community_id = pair.first;
        const auto& members = pair.second;
        
        file << community_id << " " << members.size() << " ";
        for (int node : members) {
            file << node << " ";
        }
        file << std::endl;
    }
    
    file.close();
    std::cout << "Graph with communities saved to: " << filename << std::endl;
}

Graph loadFootballGraph(){
    Graph graph;
    std::string filename = "datasets/football/football.gml";
    
    graph.loadFromFileGML(filename);
    return graph;
}

Graph loadTestGraph(){
    Graph graph;
    std::string filename = "datasets/test/test_graph.gml";
    
    graph.loadFromFileGML(filename);
    return graph;
}
// Main function with MPI initialization
int main(int argc, char* argv[]) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse command line arguments (similar to original)
    int num_ants = 30;
    int num_iterations = 200;
    double alpha = 1.0;
    double beta = 2.0;
    double rho = 0.1;
    double q0 = 0.75;
    int sync_frequency = 5;
    
    if (argc > 1) num_ants = std::stoi(argv[1]);
    if (argc > 2) num_iterations = std::stoi(argv[2]);
    if (argc > 3) alpha = std::stod(argv[3]);
    if (argc > 4) beta = std::stod(argv[4]);
    if (argc > 5) rho = std::stod(argv[5]);
    if (argc > 6) q0 = std::stod(argv[6]);
    if (argc > 7) sync_frequency = std::stoi(argv[7]);
    
    // Load graph (only on rank 0)
    Graph graph;
    
    if (rank == 0) {
        std::cout << "Loading graph..." << std::endl;
        
        // graph.loadFromFileGML("datasets/football/football.gml");
        // graph.loadFromFileGML("datasets/test/test_graph.gml");
        graph.loadSoftwareFromFileGML("datasets/software/jung-c.gml");
        // Alternative: graph.loadFromFile("datasets/DBLP/com-dblp.ungraph.txt");
        // graph = loadTestGraph();
        // graph = loadFootballGraph();
        std::cout << "Graph loaded with " << graph.getNumNodes() << " nodes and " 
                  << graph.getNumEdges() << " edges" << std::endl;
    }
    
    // Broadcast graph to all processes
    std::vector<int> graphBuffer;
    if (rank == 0) {
        graph.serializeGraph(graphBuffer);
    }
    
    // First broadcast the buffer size
    int bufferSize = graphBuffer.size();
    MPI_Bcast(&bufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Resize buffer on non-root processes and broadcast the data
    if (rank != 0) {
        graphBuffer.resize(bufferSize);
    }
    MPI_Bcast(graphBuffer.data(), bufferSize, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Deserialize graph on non-root processes
    if (rank != 0) {
        graph.deserializeGraph(graphBuffer);
    }
    
    // Create MPI ACO algorithm
    MPINodeCommunityACO aco(rank, size, graph, num_ants, num_iterations, 
                           alpha, beta, rho, q0, sync_frequency);
    
    // Get start time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run ACO
    std::unordered_map<int, std::set<int>> communities = aco.run();
    
    // Get end time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    if (rank == 0) {
        std::cout << "\nMPI Node-Community ACO completed in " 
                  << elapsed.count() << " seconds" << std::endl;
        
        // Print and save results (only on rank 0)
        printCommunities(communities);
        saveGraphWithCommunities(graph, communities, "mpi_communities.txt");
    }
    
    // Finalize MPI environment
    MPI_Finalize();
    
    return 0;
}