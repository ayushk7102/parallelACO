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
    }

    int getNumNodes() const { return numNodes; }
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
};

struct Solution {
    std::unordered_map<int, int> community; // node_id -> community_id
    double modularity;
    
    Solution() : modularity(0.0) {}
    
    // Initialize solution with each node in its own community
    explicit Solution(const std::vector<int>& nodes) : modularity(0.0) {
        for (int node : nodes) {
            community[node] = node;
        }
    }
};

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
    
    std::unordered_map<int, std::unordered_map<int, double>> pheromones;
        Solution best_solution;
    
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;
    
    void initializePheromones() {
        std::vector<int> nodes = graph.getNodeIDs();
        double initial_pheromone = 1.0 / nodes.size();
        std::cout << "Initialization started..." << std::endl;
        
        #pragma omp parallel for
        for (size_t i = 0; i < nodes.size(); i++) {
            int node = nodes[i];
            
            std::map<int, double> local_pheromones;
            for (int possible_community : nodes) {
                local_pheromones[possible_community] = initial_pheromone;
            }
            
            #pragma omp critical (pheromone_init)
            {
                for (const auto& pair : local_pheromones) {
                    pheromones[node][pair.first] = pair.second;
                }
            }
        }
        std::cout << std::endl << "Initialization complete" << std::endl;
    }
    
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
    
    Solution constructSolution(int ant_id, ThreadLocalRandom& local_rng) {
        std::vector<int> nodes = graph.getNodeIDs();
        Solution solution(nodes);
        const auto& adj_list = graph.getAdjList();
        
        local_rng.shuffle(nodes);
        
        if (ant_id > 0) {
            for (int node : nodes) {
                std::set<int> potential_communities;
                potential_communities.insert(node);
                
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
                        double pheromone;
                        #pragma omp critical (pheromone_read)
                        {
                            pheromone = pheromones[node][comm];
                        }
                        
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
                    // Exploration: probabilistic choice
                    std::vector<int> communities;
                    std::vector<double> probabilities;
                    double total = 0.0;
                    
                    for (int comm : potential_communities) {
                        double pheromone;
                        #pragma omp critical (pheromone_read)
                        {
                            pheromone = pheromones[node][comm];
                        }
                        
                        int connections = connectionsToCommunity(node, comm, solution);
                        double heuristic = std::max(0.1, static_cast<double>(connections));
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
                    solution.community[node] = selected_community;
                }
            }
        }
        solution.modularity = calculateModularity(solution);
        return solution;
    }
    
    void updatePheromones(const std::vector<Solution>& solutions) {
        std::vector<int> node_keys;
        for (const auto& pair : pheromones) {
            node_keys.push_back(pair.first);
        }

        #pragma omp parallel for
        for (size_t i = 0; i < node_keys.size(); i++) {
            int node = node_keys[i];
            
            #pragma omp critical (pheromone_update)
            {
                for (auto& comm_pair : pheromones[node]) {
                    comm_pair.second *= (1.0 - rho);
                }
            }
        }
        
        for (const auto& solution : solutions) {
            double delta = solution.modularity;
            for (const auto& pair : solution.community) {
                int node = pair.first;
                int comm = pair.second;
                #pragma omp critical (pheromone_update)
                {
                    pheromones[node][comm] += delta;
                }
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
            
            #pragma omp parallel for
            for (size_t i = 0; i < node_keys.size(); i++) {
                int node = node_keys[i];
                
                #pragma omp critical (pheromone_update)
                {
                    for (auto& comm_pair : pheromones[node]) {
                        comm_pair.second *= scale_factor;
                    }
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
                           int threads = 4)
        : graph(g), num_ants(ants), max_iterations(iterations),
          alpha(a), beta(b), rho(r), q0(q), num_threads(threads),
          rng(std::random_device{}()), dist(0.0, 1.0) {
        
        omp_set_num_threads(num_threads);
        
        initializePheromones();
        best_solution = Solution(graph.getNodeIDs());
        best_solution.modularity = calculateModularity(best_solution);
    }
    
    std::unordered_map<int, std::set<int>> run() {
        std::cout << "Running Parallel Node-Community ACO with:" << std::endl;
        std::cout << "  - " << num_ants << " ants" << std::endl;
        std::cout << "  - " << max_iterations << " iterations" << std::endl;
        std::cout << "  - " << num_threads << " threads" << std::endl;
        std::cout << "  - Alpha (pheromone influence): " << alpha << std::endl;
        std::cout << "  - Beta (heuristic influence): " << beta << std::endl;
        std::cout << "  - Rho (evaporation rate): " << rho << std::endl;
        std::cout << "  - q0 (exploitation probability): " << q0 << std::endl;
        
        std::vector<ThreadLocalRandom> thread_rngs(num_threads);
        for (int i = 0; i < num_threads; i++) {
            thread_rngs[i] = ThreadLocalRandom(std::random_device{}() + i);
        }
        
        for (int iter = 0; iter < max_iterations; iter++) {
            std::vector<Solution> ant_solutions(num_ants);
            
            #pragma omp parallel for
            for (int ant = 0; ant < num_ants; ant++) {
                int thread_id = omp_get_thread_num();
                ant_solutions[ant] = constructSolution(ant, thread_rngs[thread_id]);
                
                #pragma omp critical (best_solution)
                {
                    if (ant_solutions[ant].modularity > best_solution.modularity) {
                        best_solution = ant_solutions[ant];
                    }
                }
            }
            
            updatePheromones(ant_solutions);
            
            // Print progress every iteration
            // auto communities = convertToCommunityMap(best_solution);
            // if (iter % 10 == 0 or iter == max_iterations - 1) {
            //     std::cout << "Iteration " << iter << ": " 
            //                         << communities.size() << " communities, "
            //                         << "modularity = " << best_solution.modularity << std::endl;
            // }
        }
        
        auto communities = convertToCommunityMap(best_solution);
        std::cout << "Final result: " << communities.size() 
                  << " communities, modularity = " << best_solution.modularity << std::endl;
        
        return communities;
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

Graph loadFootballGraph() {
    Graph graph;
    std::string filename = "datasets/football/football.gml";
    
    graph.loadFromFileGML(filename);
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
    
    if (argc > 1) num_ants = std::stoi(argv[1]);
    if (argc > 2) num_iterations = std::stoi(argv[2]);
    if (argc > 3) alpha = std::stod(argv[3]);
    if (argc > 4) beta = std::stod(argv[4]);
    if (argc > 5) rho = std::stod(argv[5]);
    if (argc > 6) q0 = std::stod(argv[6]);
    if (argc > 7) num_threads = std::stoi(argv[7]);
    
    std::cout << "Loading graph..." << std::endl;
    Graph graph = loadFootballGraph();
    
    std::cout << "Graph loaded with " << graph.getNumNodes() << " nodes and " 
              << graph.getNumEdges() << " edges" << std::endl;
    
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    std::cout << "Using " << omp_get_max_threads() << " threads" << std::endl;
    
    ParallelNodeCommunityACO aco(graph, num_ants, num_iterations, alpha, beta, rho, q0, num_threads);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::unordered_map<int, std::set<int>> communities = aco.run();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "\nParallel Node-Community ACO completed in " 
              << elapsed.count() << " seconds" << std::endl;
    
    printCommunities(communities);
    
    saveGraphWithCommunities(graph, communities, "parallel_communities.txt");
    
    return 0;
}