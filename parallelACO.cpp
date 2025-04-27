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

    void loadSoftwareFromFileGML(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        // Assuming we have a data structure to track nodes
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
};

// Structure to represent a solution (community assignment)
struct Solution {
    std::unordered_map<int, int> community; // node_id -> community_id
    double modularity;
    
    Solution() : modularity(0.0) {}
    
    // Initialize solution with each node in its own community
    explicit Solution(const std::vector<int>& nodes) : modularity(0.0) {
        for (int node : nodes) {
            community[node] = node; // Initially node ID is community ID
        }
    }
};

// Thread-local random generator for parallel execution
class ThreadLocalRandom {
private:
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;

public:
    ThreadLocalRandom() : rng(std::random_device()()), dist(0.0, 1.0) {}
    
    explicit ThreadLocalRandom(unsigned int seed) : rng(seed), dist(0.0, 1.0) {}
    
    double random() {
        return dist(rng);
    }
    
    // Shuffle a vector
    template<typename T>
    void shuffle(std::vector<T>& v) {
        std::shuffle(v.begin(), v.end(), rng);
    }
};

class ParallelNodeCommunityACO {
private:
    const Graph& graph;
    int num_ants;
    int max_iterations;
    double alpha;     // Pheromone influence
    double beta;      // Heuristic influence
    double rho;       // Evaporation rate
    double q0;        // Probability of exploitation vs exploration
    int num_threads;  // Number of OpenMP threads
    int chunk_size;   // Chunk size for dynamic scheduling
    bool use_local_pheromones; // Use thread-local pheromone copies
    
    // Pheromone matrix: pheromone[node][community] = pheromone level
    std::unordered_map<int, std::unordered_map<int, double>> pheromones;
    
    // Best solution found so far
    Solution best_solution;
    
    // Master random number generator
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;
    
    // Initialize pheromone values with optimized approach
    void initializePheromones() {
        std::vector<int> nodes = graph.getNodeIDs();
        double initial_pheromone = 1.0 / nodes.size();
        std::cout << "Initialization started..." << std::endl;
        
        // Calculate chunk size for better load balancing
        const int CHUNK_SIZE = std::max(1, static_cast<int>(nodes.size() / (omp_get_max_threads() * 2)));
        
        // This can be parallelized with dynamic scheduling
        #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
        for (size_t i = 0; i < nodes.size(); i++) {
            int node = nodes[i];
            
            // Create a batch of pheromone values locally to reduce contention
            std::unordered_map<int, double> local_pheromones;
            for (int possible_community : nodes) {
                local_pheromones[possible_community] = initial_pheromone;
            }
            
            // Single critical section per node instead of per community
            #pragma omp critical (pheromone_init)
            {
                pheromones[node] = std::move(local_pheromones);
            }
        }
        
        std::cout << std::endl << "Initialization complete" << std::endl;
    }
    
    // Print statistics about the pheromone matrix
    void printPheromoneStats() {
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
    
    // Optimized solution construction with thread-local pheromone copies
    Solution constructSolution(int ant_id, ThreadLocalRandom& local_rng) {
        std::vector<int> nodes = graph.getNodeIDs();
        Solution solution(nodes);
        const auto& adj_list = graph.getAdjList();
        
        // Shuffle nodes to process them in random order
        local_rng.shuffle(nodes);
        
        // First ant uses initial solution, others construct new ones
        if (ant_id > 0) {
            // Thread-local pheromone copy to reduce critical sections
            std::unordered_map<int, std::unordered_map<int, double>> local_pheromones;
            
            // Only create local copy if the flag is set
            if (use_local_pheromones) {
                // Copy pheromones just once at the beginning
                #pragma omp critical (pheromone_read_bulk)
                {
                    for (int node : nodes) {
                        local_pheromones[node] = pheromones[node];
                    }
                }
            }
            
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
                double q = local_rng.random();
                
                if (q < q0) {
                    // Exploitation: choose best community by deterministic rule
                    int best_community = node;
                    double best_value = 0.0;
                    
                    for (int comm : potential_communities) {
                        // Pheromone level - use local copy if available
                        double pheromone;
                        if (use_local_pheromones) {
                            pheromone = local_pheromones[node][comm];
                        } else {
                            #pragma omp critical (pheromone_read)
                            {
                                pheromone = pheromones[node][comm];
                            }
                        }
                        
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
                        // Pheromone level - use local copy if available
                        double pheromone;
                        if (use_local_pheromones) {
                            pheromone = local_pheromones[node][comm];
                        } else {
                            #pragma omp critical (pheromone_read)
                            {
                                pheromone = pheromones[node][comm];
                            }
                        }
                        
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
                    double r = local_rng.random();
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
    
    // Optimized pheromone update with reduced critical sections
    void updatePheromones(const std::vector<Solution>& solutions) {
        // Get all node IDs first to avoid race conditions in key iteration
        std::vector<int> all_nodes;
        for (const auto& pair : pheromones) {
            all_nodes.push_back(pair.first);
        }
        
        // Get all community IDs that have pheromones
        std::set<int> all_communities;
        for (const auto& node_map : pheromones) {
            for (const auto& comm_pair : node_map.second) {
                all_communities.insert(comm_pair.first);
            }
        }
        
        // STEP 1: Evaporation - parallelize over nodes
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (size_t i = 0; i < all_nodes.size(); i++) {
            int node = all_nodes[i];
            
            // Create a thread-local copy to update
            std::unordered_map<int, double> local_pheromones;
            
            // Copy current pheromones with evaporation applied
            #pragma omp critical (pheromone_read)
            {
                for (const auto& comm_pair : pheromones[node]) {
                    local_pheromones[comm_pair.first] = comm_pair.second * (1.0 - rho);
                }
            }
            
            // STEP 2: Add new pheromones from solutions (still thread-local)
            for (const auto& solution : solutions) {
                if (solution.community.find(node) != solution.community.end()) {
                    int comm = solution.community.at(node);
                    local_pheromones[comm] += solution.modularity;
                }
            }
            
            // STEP 3: Update global pheromones (single critical section per node)
            #pragma omp critical (pheromone_write)
            {
                for (const auto& comm_pair : local_pheromones) {
                    pheromones[node][comm_pair.first] = comm_pair.second;
                }
            }
        }
        
        // Normalize pheromones - single parallel region with reduced critical sections
        double max_pheromone = 0.0;
        
        // Find max pheromone value in parallel
        #pragma omp parallel
        {
            double thread_max = 0.0;
            
            #pragma omp for nowait
            for (size_t i = 0; i < all_nodes.size(); i++) {
                int node = all_nodes[i];
                for (const auto& comm_pair : pheromones[node]) {
                    thread_max = std::max(thread_max, comm_pair.second);
                }
            }
            
            // Combine thread-local max values
            #pragma omp critical (max_reduction)
            {
                max_pheromone = std::max(max_pheromone, thread_max);
            }
        }
        
        // If normalization needed, apply in parallel
        if (max_pheromone > 10.0) {
            double scale_factor = 5.0 / max_pheromone;
            
            #pragma omp parallel for schedule(dynamic, chunk_size)
            for (size_t i = 0; i < all_nodes.size(); i++) {
                int node = all_nodes[i];
                
                // Make local changes first
                std::unordered_map<int, double> scaled_values;
                for (const auto& comm_pair : pheromones[node]) {
                    scaled_values[comm_pair.first] = comm_pair.second * scale_factor;
                }
                
                // Single critical section per node
                #pragma omp critical (pheromone_scale)
                {
                    for (const auto& comm_pair : scaled_values) {
                        pheromones[node][comm_pair.first] = comm_pair.second;
                    }
                }
            }
        }
    }
    
    // Calculate modularity of a solution - optimized with precalculated community degrees
    double calculateModularity(const Solution& solution) {
        const auto& adj_list = graph.getAdjList();
        double m = graph.getNumEdges(); // Total number of edges
        double q = 0.0;
        
        // Pre-calculate all community degrees to avoid repeated computation
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
    ParallelNodeCommunityACO(const Graph& g, int ants = 20, int iterations = 100, 
                           double a = 1.0, double b = 2.0, double r = 0.1, double q = 0.9,
                           int threads = 4, int chunk = 1, bool local_pheromones = true)
        : graph(g), num_ants(ants), max_iterations(iterations),
          alpha(a), beta(b), rho(r), q0(q), num_threads(threads),
          chunk_size(chunk), use_local_pheromones(local_pheromones),
          rng(std::random_device{}()), dist(0.0, 1.0) {
        
        // Set the number of OpenMP threads
        omp_set_num_threads(num_threads);
        
        initializePheromones();
        best_solution = Solution(graph.getNodeIDs());
        best_solution.modularity = calculateModularity(best_solution);
    }
    
    std::unordered_map<int, std::set<int>> run() {
        std::cout << "Running Optimized Parallel Node-Community ACO with:" << std::endl;
        std::cout << "  - " << num_ants << " ants" << std::endl;
        std::cout << "  - " << max_iterations << " iterations" << std::endl;
        std::cout << "  - " << num_threads << " threads" << std::endl;
        std::cout << "  - Alpha (pheromone influence): " << alpha << std::endl;
        std::cout << "  - Beta (heuristic influence): " << beta << std::endl;
        std::cout << "  - Rho (evaporation rate): " << rho << std::endl;
        std::cout << "  - q0 (exploitation probability): " << q0 << std::endl;
        std::cout << "  - Chunk size: " << chunk_size << std::endl;
        std::cout << "  - Using local pheromones: " << (use_local_pheromones ? "yes" : "no") << std::endl;
        
        // Create thread-local random generators
        std::vector<ThreadLocalRandom> thread_rngs(num_threads);
        for (int i = 0; i < num_threads; i++) {
            thread_rngs[i] = ThreadLocalRandom(std::random_device{}() + i);
        }
        
        auto total_start_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_construction_time(0);

        for (int iter = 0; iter < max_iterations; iter++) {
            std::vector<Solution> ant_solutions(num_ants);

            auto construction_start_time = std::chrono::high_resolution_clock::now();

            
            // Parallel solution construction with dynamic scheduling for better load balancing
            #pragma omp parallel for schedule(dynamic, chunk_size)
            for (int ant = 0; ant < num_ants; ant++) {
                // Get the thread ID for thread-local RNG
                int thread_id = omp_get_thread_num();
                ant_solutions[ant] = constructSolution(ant, thread_rngs[thread_id]);
                
                // Local search to improve solution - optional, can be disabled for speed
                // ant_solutions[ant] = localSearch(ant_solutions[ant]);
                
                // Update best solution if improved - needs to be thread-safe
                #pragma omp critical (best_solution)
                {
                    if (ant_solutions[ant].modularity > best_solution.modularity) {
                        best_solution = ant_solutions[ant];
                    }
                }
            }

            auto construction_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> construction_elapsed = construction_end_time - construction_start_time;
            total_construction_time += construction_elapsed;
            
            // Update pheromones based on solutions
            updatePheromones(ant_solutions);

            std::cout<<"Iteration "<<iter<<" done\n"<<std::endl;
        }

        auto total_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_elapsed = total_end_time - total_start_time;

        std::cout << "\nTotal solution construction time: " << total_construction_time.count()  << std::endl;
        // Final result
        auto communities = convertToCommunityMap(best_solution);
        std::cout << "Final result: " << communities.size() 
                  << " communities, modularity = " << best_solution.modularity << std::endl;
        
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

Graph loadFootballGraph() {
    Graph graph;
    std::string filename = "datasets/football/football.gml";
    
    graph.loadFromFileGML(filename);
    return graph;
}


Graph loadSoftwareGraph() {
    Graph graph;
    std::string filename = "datasets/software/jung-c.gml";
    
    graph.loadSoftwareFromFileGML(filename);
    return graph;
}

Graph loadDBLPGraph() {
    Graph graph;
    std::string filename = "datasets/DBLP/com-dblp.ungraph.txt";
    
    graph.loadFromFile(filename);
    return graph;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    int num_ants = 30;
    int num_iterations = 200;
    double alpha = 1.0;
    double beta = 2.0;
    double rho = 0.1;
    double q0 = 0.9;
    int num_threads = omp_get_max_threads(); // Default to max available threads
    int chunk_size = 1; // Default chunk size for dynamic scheduling
    bool use_local_pheromones = true; // Default to using thread-local pheromones
    
    if (argc > 1) num_ants = std::stoi(argv[1]);
    if (argc > 2) num_iterations = std::stoi(argv[2]);
    if (argc > 3) alpha = std::stod(argv[3]);
    if (argc > 4) beta = std::stod(argv[4]);
    if (argc > 5) rho = std::stod(argv[5]);
    if (argc > 6) q0 = std::stod(argv[6]);
    if (argc > 7) num_threads = std::stoi(argv[7]);
    if (argc > 8) chunk_size = std::stoi(argv[8]);
    if (argc > 9) use_local_pheromones = std::stoi(argv[9]) > 0;
    
    // Load graph
    std::cout << "Loading graph..." << std::endl;
    Graph graph = loadFootballGraph(); // Change to loadDBLPGraph() for DBLP dataset
    
    std::cout << "Graph loaded with " << graph.getNumNodes() << " nodes and " 
              << graph.getNumEdges() << " edges" << std::endl;
    
    // Set number of threads for OpenMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    std::cout << "Using " << omp_get_max_threads() << " threads" << std::endl;
    
    // Create parallel ACO algorithm with optimized parameters
    ParallelNodeCommunityACO aco(graph, num_ants, num_iterations, alpha, beta, rho, q0, 
                                num_threads, chunk_size, use_local_pheromones);
    
    // Get start time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run ACO
    std::unordered_map<int, std::set<int>> communities = aco.run();
    
    // Get end time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "\nOptimized Parallel Node-Community ACO completed in " 
              << elapsed.count() << " seconds" << std::endl;
    
    // Print communities
    printCommunities(communities);
    
    // Save results
    saveGraphWithCommunities(graph, communities, "optimized_parallel_communities.txt");
    
    return 0;
}