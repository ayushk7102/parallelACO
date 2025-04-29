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
    std::map<int, bool> nodes;
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
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        if (line.empty()) continue; 
        std::istringstream iss(line);
        iss >> token;
        if (token == "graph") {
            inGraph = true;
        } else if (token == "directed") {
            int directed;
            iss >> directed;
        } else if (token == "node") {
            inNode = true;
            inEdge = false;
        } else if (token == "edge") {
            inNode = false;
            inEdge = true;
        } else if (token == "[") {
            continue;
        } else if (token == "]") {
            if (inNode) {
                inNode = false;
            } else if (inEdge) {
                if (sourceNode != -1 && targetNode != -1) {
                    addEdge(sourceNode, targetNode);
                    edgeCount++;
                    sourceNode = -1;
                    targetNode = -1;
                }
                inEdge = false;
            } else if (inGraph) {
                inGraph = false;
            }
        } else if (inNode && token == "id") {
            iss >> currentNodeId;
            nodes[currentNodeId] = true;
        } else if (inEdge && token == "source") {
            iss >> sourceNode;
        } else if (inEdge && token == "target") {
            iss >> targetNode;
        }
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
            line.erase(0, line.find_first_not_of(" \t"));
            if (line.empty()) continue;
            
            if (line == "graph") {
                continue;
            } else if (line == "[") {
                if (!inGraph) {
                    inGraph = true;
                    continue;
                } else if (!inNode && !inEdge && line == "[") {
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
            
            if (line == "node") {
                inNode = true;
                continue;
            }
            
            if (line == "edge") {
                inEdge = true;
                continue;
            }
            
            if (inNode && line.find("id ") == 0) {
                std::istringstream iss(line.substr(3));
                iss >> currentNodeId;
                continue;
            }
            
            if (inEdge && line.find("source ") == 0) {
                std::istringstream iss(line.substr(7));
                iss >> sourceNode;
                continue;
            }
            
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
            if (line.empty() || line[0] == '#') {
                if (line.find("Nodes:") != std::string::npos && 
                    line.find("Edges:") != std::string::npos) {
                    std::istringstream iss(line);
                    std::string token;
                    iss >> token; 
                    iss >> token >> numNodes; 
                    iss >> token >> numEdges; 
                }
                continue;
            }

            if (!headerProcessed) {
                headerProcessed = true;
                continue;
            }

            std::istringstream iss(line);
            int from, to;
            if (iss >> from >> to) {
                addEdge(from, to);
            }
        }

        file.close();
        std::cout << "Graph loaded successfully!" << std::endl;
        std::cout << "Nodes: " << numNodes << ", Edges: " << numEdges / 2 << std::endl;
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
    
    void serializeGraph(std::vector<int>& buffer) {
        buffer.clear();
        
        buffer.push_back(numNodes);
        buffer.push_back(numEdges);
        
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

struct Solution {
    std::unordered_map<int, int> community; 
    double modularity;
    
    Solution() : modularity(0.0) {}
    
    explicit Solution(const std::vector<int>& nodes) : modularity(0.0) {
        for (int node : nodes) {
            community[node] = node; 
        }
    }
    
    void serializeSolution(std::vector<int>& intBuffer, std::vector<double>& doubleBuffer) {
        intBuffer.clear();
        doubleBuffer.clear();
        
        doubleBuffer.push_back(modularity);
        
        for (const auto& pair : community) {
            intBuffer.push_back(pair.first);
            intBuffer.push_back(pair.second);
        }
    }
    
    void deserializeSolution(const std::vector<int>& intBuffer, const std::vector<double>& doubleBuffer) {
        community.clear();
        
        modularity = doubleBuffer[0];
        
        for (size_t i = 0; i < intBuffer.size(); i += 2) {
            int nodeID = intBuffer[i];
            int communityID = intBuffer[i + 1];
            community[nodeID] = communityID;
        }
    }
};

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
    
    int rank;
    int size;
    int ants_per_process;
    
    std::unordered_map<int, std::unordered_map<int, double>> pheromones;
    
    Solution best_local_solution;
    
    Solution best_global_solution;
    
    RandomGenerator rng;
    
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
    
    Solution constructSolution(int ant_id) {
        std::vector<int> nodes = graph.getNodeIDs();
        Solution solution(nodes);
        const auto& adj_list = graph.getAdjList();
        
        rng.shuffle(nodes);
        
        if (ant_id > 0) {
            for (int node : nodes) {
                std::set<int> potential_communities;
                potential_communities.insert(node);
                
                if (adj_list.find(node) != adj_list.end()) {
                    for (int neighbor : adj_list.at(node)) {
                        potential_communities.insert(solution.community[neighbor]);
                    }
                }
                
                double q = rng.random();
                
                if (q < q0) {
                    int best_community = node;
                    double best_value = 0.0;
                    
                    for (int comm : potential_communities) {
                        double pheromone = pheromones[node][comm];
                        
                        int connections = connectionsToCommunity(node, comm, solution);
                        double heuristic = std::max(0.1, static_cast<double>(connections));
                        double value = std::pow(pheromone, alpha) * std::pow(heuristic, beta);
                        
                        if (value > best_value) {
                            best_value = value;
                            best_community = comm;
                        }
                    }
                    solution.community[node] = best_community;
                }
                else {
                    // Exploration:
                    std::vector<int> communities;
                    std::vector<double> probabilities;
                    double total = 0.0;
                    
                    for (int comm : potential_communities) {
                        double pheromone = pheromones[node][comm];
                        
                        int connections = connectionsToCommunity(node, comm, solution);
                        double heuristic = std::max(0.1, static_cast<double>(connections));
                        double value = std::pow(pheromone, alpha) * std::pow(heuristic, beta);
                        
                        communities.push_back(comm);
                        probabilities.push_back(value);
                        total += value;
                    }
                    
                    if (total > 0) {
                        for (auto& p : probabilities) {
                            p /= total;
                        }
                    }
                    else {
                        for (auto& p : probabilities) {
                            p = 1.0 / probabilities.size();
                        }
                    }
                    
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
                    
                    solution.community[node] = selected_community;
                }
            }
        }
        
        solution.modularity = calculateModularity(solution);
        
        return solution;
    }
    
    void updatePheromones(const std::vector<Solution>& solutions) {
        for (auto& node_map : pheromones) {
            for (auto& comm_pair : node_map.second) {
                comm_pair.second *= (1.0 - rho);
            }
        }
        
        for (const auto& solution : solutions) {
            double delta = solution.modularity;
            
            for (const auto& pair : solution.community) {
                int node = pair.first;
                int comm = pair.second;
                pheromones[node][comm] += delta;
            }
        }
        
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
    
    double calculateModularity(const Solution& solution) {
        const auto& adj_list = graph.getAdjList();
        double m = graph.getNumEdges(); 
        double q = 0.0;
        
        std::unordered_map<int, double> community_degrees;
        for (const auto& pair : adj_list) {
            int node = pair.first;
            
            if (solution.community.find(node) == solution.community.end()) {
                continue;
            }
            
            int comm = solution.community.at(node);
            if (community_degrees.find(comm) == community_degrees.end()) {
                community_degrees[comm] = 0.0;
            }
            community_degrees[comm] += pair.second.size();
        }
        
        for (const auto& pair : adj_list) {
            int i = pair.first;
            
            if (solution.community.find(i) == solution.community.end()) {
                continue;
            }
            
            int c_i = solution.community.at(i);
            
            for (int j : pair.second) {
                if (i < j) {
                    if (solution.community.find(j) == solution.community.end()) {
                        continue;
                    }
                    int c_j = solution.community.at(j);
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
    
    void synchronizeBestSolutions() {
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
        
        rng = RandomGenerator(std::random_device{}() + rank);
        
        ants_per_process = num_ants / size;
        if (rank < num_ants % size) {
            ants_per_process++;
        }
        
        initializePheromones();
        
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
        double computationTime = 0.0;
        for (int iter = 0; iter < max_iterations; iter++) {
            std::vector<Solution> ant_solutions(ants_per_process);
            
            for (int ant = 0; ant < ants_per_process; ant++) {
                auto start_time = std::chrono::high_resolution_clock::now();

                ant_solutions[ant] = constructSolution(ant);
                std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start_time;
                computationTime += elapsed.count();

                if (ant_solutions[ant].modularity > best_local_solution.modularity) {
                    best_local_solution = ant_solutions[ant];
                }
            }
            
            updatePheromones(ant_solutions);
            
            // if (iter % sync_frequency == 0 || iter == max_iterations - 1) {
            //     synchronizeBestSolutions();
                
            //     if (rank == 0 && (iter % 10 == 0 || iter == max_iterations - 1)) {
            //         auto communities = convertToCommunityMap(best_global_solution);
            //         std::cout << "Iteration " << iter << ": " 
            //                   << communities.size() << " communities, "
            //                   << "modularity = " << best_global_solution.modularity << std::endl;
                    
            //         if (iter % 10 == 0) {
            //             printPheromoneStats();
            //         }
            //     }
            // }
        }
        
        synchronizeBestSolutions();
        
        if (rank == 0) {
            auto communities = convertToCommunityMap(best_global_solution);
            std::cout << "-----------------TOTAL COMPUTATION TIME = " << computationTime << " seconds "<<"\n";
            std::cout << "Final result: " << communities.size() 
                      << " communities, modularity = " << best_global_solution.modularity << std::endl;
            return communities;
        } else {
            return std::unordered_map<int, std::set<int>>();
        }
        
    }
};

void printCommunities(const std::unordered_map<int, std::set<int>>& communities) {

    std::vector<std::pair<int, std::set<int>>> sorted_communities;
    for (const auto& pair : communities) {
        sorted_communities.push_back(pair);
    }
    
    std::sort(sorted_communities.begin(), sorted_communities.end(),
              [](const auto& a, const auto& b) { return a.second.size() > b.second.size(); });
    
    std::cout << "Detected " << communities.size() << " communities:" << std::endl;
    

    int count = 0;
    for (const auto& pair : sorted_communities) {
        std::cout << "Community " << pair.first << " (size: " << pair.second.size() << "): ";
        
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
    
    std::unordered_map<int, int> node_to_community;
    for (const auto& pair : communities) {
        int community_id = pair.first;
        for (int node : pair.second) {
            node_to_community[node] = community_id;
        }
    }
    
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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
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
    
    Graph graph;
    
    if (rank == 0) {
        std::cout << "Loading graph..." << std::endl;
        
        // graph.loadFromFileGML("datasets/football/football.gml");
        graph.loadSoftwareFromFileGML("datasets/metabolic/celegans_metabolic.gml");
        // graph.loadSoftwareFromFileGML("datasets/software/jung-c.gml");
        // Alternative: graph.loadFromFile("datasets/DBLP/com-dblp.ungraph.txt");
        // graph = loadTestGraph();
        // graph = loadFootballGraph();
        std::cout << "Graph loaded with " << graph.getNumNodes() << " nodes and " 
                  << graph.getNumEdges() << " edges" << std::endl;
    }
    
    std::vector<int> graphBuffer;
    if (rank == 0) {
        graph.serializeGraph(graphBuffer);
    }
    
    int bufferSize = graphBuffer.size();
    MPI_Bcast(&bufferSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        graphBuffer.resize(bufferSize);
    }
    MPI_Bcast(graphBuffer.data(), bufferSize, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        graph.deserializeGraph(graphBuffer);
    }
    
    MPINodeCommunityACO aco(rank, size, graph, num_ants, num_iterations, 
                           alpha, beta, rho, q0, sync_frequency);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    std::unordered_map<int, std::set<int>> communities = aco.run();
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end_time - start_time;
    
    if (rank == 0) {
        std::cout << "\nMPI Node-Community ACO completed in " 
                  << elapsed.count() << " seconds" << std::endl;
        
        printCommunities(communities);
        saveGraphWithCommunities(graph, communities, "mpi_communities.txt");
    }
    
    MPI_Finalize();
    
    return 0;
}