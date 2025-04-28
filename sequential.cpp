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
// #include "/opt/homebrew/Cellar/libomp/20.1.2/include/omp.h"

class Graph {
private:
    std::unordered_map<int, std::vector<int> > adjList;
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
    }

    int getNumNodes() const { return numNodes; }
    int getNumEdges() const { return numEdges / 2; }
    
    const std::unordered_map<int, std::vector<int> >& getAdjList() const {
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

class NodeCommunityACO {
private:
    const Graph& graph;
    int num_ants;
    int max_iterations;
    double alpha;     // Pheromone influence
    double beta;      // Heuristic influence
    double rho;       // Evaporation rate
    double q0;        // Probability of exploitation vs exploration
    
    // Pheromone matrix: pheromone[node][community] = pheromone level
    std::unordered_map<int, std::unordered_map<int, double>> pheromones;
    
    // Best solution found so far
    Solution best_solution;
    
    // Random number generator
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;
    
    // Initialize pheromone values
    void initializePheromones() {
        std::vector<int> nodes = graph.getNodeIDs();
        double initial_pheromone = 1.0 / nodes.size();
        for (int node : nodes) {
            for (int possible_community : nodes) {
                pheromones[node][possible_community] = initial_pheromone;
            }
            // if(node % 1000 == 0) {
            //     std::cout<<"on node: "<<node<<"\n";
            // }
        }
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
    
    // Construct a solution using ant colony principles
    Solution constructSolution(int ant_id) {
        std::vector<int> nodes = graph.getNodeIDs();
        Solution solution(nodes);
        const auto& adj_list = graph.getAdjList();
        
        // Shuffle nodes to process them in random order
        std::shuffle(nodes.begin(), nodes.end(), rng);
        
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
                double q = dist(rng);
                
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
                    double r = dist(rng);
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
        
        // Apply local search to improve the solution
        // solution = localSearch(solution);
        
        // Calculate solution modularity
        solution.modularity = calculateModularity(solution);
        return solution;
    }
    
    // Update pheromone levels based on solutions
    void updatePheromones(const std::vector<Solution>& solutions) {
        // Evaporation
        for (auto& node_map : pheromones) {
            for (auto& comm_val : node_map.second) {
                comm_val.second *= (1.0 - rho);
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
                for (auto& comm_val : node_map.second) {
                    comm_val.second *= scale_factor;
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
    NodeCommunityACO(const Graph& g, int ants = 20, int iterations = 100, 
                     double a = 1.0, double b = 2.0, double r = 0.1, double q = 0.9)
        : graph(g), num_ants(ants), max_iterations(iterations),
          alpha(a), beta(b), rho(r), q0(q), 
          rng(std::random_device{}()), dist(0.0, 1.0) {
        
        initializePheromones();
        best_solution = Solution(graph.getNodeIDs());
        best_solution.modularity = calculateModularity(best_solution);
    }
    
    std::unordered_map<int, std::set<int>> run() {
        std::cout << "Running Node-Community ACO with:" << std::endl;
        std::cout << "  - " << num_ants << " ants" << std::endl;
        std::cout << "  - " << max_iterations << " iterations" << std::endl;
        std::cout << "  - Alpha (pheromone influence): " << alpha << std::endl;
        std::cout << "  - Beta (heuristic influence): " << beta << std::endl;
        std::cout << "  - Rho (evaporation rate): " << rho << std::endl;
        std::cout << "  - q0 (exploitation probability): " << q0 << std::endl;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            std::vector<Solution> ant_solutions(num_ants);
            // Each ant constructs a solution
            for (int ant = 0; ant < num_ants; ant++) {
                ant_solutions[ant] = constructSolution(ant);
                
                // Update best solution if improved
                if (ant_solutions[ant].modularity > best_solution.modularity) {
                    best_solution = ant_solutions[ant];
                }
            }
            
            // Update pheromones based on solutions
            updatePheromones(ant_solutions);
            
            // Print progress every 10 iterations or at the end
            if (iter % 10 == 0 || iter == max_iterations - 1) {
                // Convert the current best solution to community format
                // auto communities = convertToCommunityMap(best_solution);
                
                // std::cout << "Iteration " << iter << ": " 
                //           << communities.size() << " communities, "
                //           << "modularity = " << best_solution.modularity << std::endl;
                
                // if (iter % 50 == 0) {
                //     printPheromoneStats();
                // }
                std::cout<<iter<<std::endl;
            }
        }
        
        // Final result
        auto communities = convertToCommunityMap(best_solution);
        std::cout << "Final result: " << communities.size() 
                  << " communities, modularity = " << best_solution.modularity << std::endl;
        
        return communities;
    }
};

class GirvanNewman {
private:
    const Graph& graph;
    
    // Calculate betweenness of all edges in the graph
    std::map<std::pair<int, int>, double> calculateEdgeBetweenness() {
        std::map<std::pair<int, int>, double> betweenness;
        std::vector<int> nodes = graph.getNodeIDs();
        const auto& adjList = graph.getAdjList();
        
        // Initialize betweenness values to 0
        for (const auto& pair : adjList) {
            int node = pair.first;
            for (int neighbor : pair.second) {
                if (node < neighbor) { // Count each edge only once
                    betweenness[{node, neighbor}] = 0.0;
                }
            }
        }
        
        // For each node as source
        for (int source : nodes) {
            // BFS to calculate shortest paths
            std::queue<int> q;
            std::map<int, int> distance;
            std::map<int, double> pathCount;
            std::map<int, std::vector<int>> predecessors;
            
            // Initialize
            for (int node : nodes) {
                distance[node] = -1;
                pathCount[node] = 0.0;
            }
            
            distance[source] = 0;
            pathCount[source] = 1.0;
            q.push(source);
            
            // BFS phase
            while (!q.empty()) {
                int current = q.front();
                q.pop();
                
                // Process neighbors
                for (int neighbor : adjList.at(current)) {
                    // If not visited yet
                    if (distance[neighbor] == -1) {
                        distance[neighbor] = distance[current] + 1;
                        q.push(neighbor);
                    }
                    
                    // If this neighbor is further away, it's on a shortest path
                    if (distance[neighbor] == distance[current] + 1) {
                        pathCount[neighbor] += pathCount[current];
                        predecessors[neighbor].push_back(current);
                    }
                }
            }
            
            // Calculate edge betweenness
            std::map<int, double> nodeDependency;
            for (int node : nodes) {
                nodeDependency[node] = 0.0;
            }
            
            // Visit nodes in order of decreasing distance from source
            std::vector<int> stack;
            for (int node : nodes) {
                if (node != source && distance[node] != -1) {
                    stack.push_back(node);
                }
            }
            
            // Sort by distance (decreasing)
            std::sort(stack.begin(), stack.end(), [&distance](int a, int b) {
                return distance[a] > distance[b];
            });
            
            // Process nodes in order
            for (int node : stack) {
                // Each predecessor gets credit proportional to the number of paths
                for (int pred : predecessors[node]) {
                    double edgeCredit = (pathCount[pred] / pathCount[node]) * (1.0 + nodeDependency[node]);
                    
                    // Add to edge betweenness (ensure canonical edge ordering)
                    if (pred < node) {
                        betweenness[{pred, node}] += edgeCredit;
                    } else {
                        betweenness[{node, pred}] += edgeCredit;
                    }
                    
                    nodeDependency[pred] += edgeCredit;
                }
            }
        }
        
        return betweenness;
    }
    
    // Find connected components in a graph
    std::vector<std::set<int>> findConnectedComponents(
            const std::unordered_map<int, std::vector<int>>& currentGraph) {
        std::vector<std::set<int>> components;
        std::set<int> visited;
        
        // Get all nodes
        std::vector<int> nodes;
        for (const auto& pair : currentGraph) {
            nodes.push_back(pair.first);
        }
        
        // Find components using BFS
        for (int node : nodes) {
            if (visited.find(node) == visited.end()) {
                // Start a new component
                std::set<int> component;
                std::queue<int> q;
                
                q.push(node);
                visited.insert(node);
                component.insert(node);
                
                while (!q.empty()) {
                    int current = q.front();
                    q.pop();
                    
                    // Check all neighbors
                    if (currentGraph.find(current) != currentGraph.end()) {
                        for (int neighbor : currentGraph.at(current)) {
                            if (visited.find(neighbor) == visited.end()) {
                                visited.insert(neighbor);
                                component.insert(neighbor);
                                q.push(neighbor);
                            }
                        }
                    }
                }
                
                components.push_back(component);
            }
        }
        
        return components;
    }
    
    // Calculate modularity for a given community assignment
    double calculateModularity(const std::vector<std::set<int>>& communities) {
        const auto& adjList = graph.getAdjList();
        double m = graph.getNumEdges(); // Total number of edges
        double q = 0.0;
        
        // Create community lookup map for faster access
        std::unordered_map<int, int> communityMap;
        for (size_t i = 0; i < communities.size(); i++) {
            for (int node : communities[i]) {
                communityMap[node] = i;
            }
        }
        
        // For each edge
        for (const auto& pair : adjList) {
            int i = pair.first;
            
            if (communityMap.find(i) == communityMap.end()) {
                continue;
            }
            
            int c_i = communityMap[i];
            
            for (int j : pair.second) {
                // Process each edge once (for undirected graph)
                if (i < j) {
                    if (communityMap.find(j) == communityMap.end()) {
                        continue;
                    }
                    
                    int c_j = communityMap[j];
                    
                    // Same community contribution
                    if (c_i == c_j) {
                        double expected = (pair.second.size() * adjList.at(j).size()) / (2.0 * m);
                        q += 1.0 - expected;
                    }
                }
            }
        }
        
        q /= (2.0 * m);
        return q;
    }

public:
    GirvanNewman(const Graph& g) : graph(g) {}
    
    // Run the Girvan-Newman algorithm to find communities
    std::unordered_map<int, std::set<int>> findCommunities(int maxRemovals = -1) {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::cout << "Running Girvan-Newman algorithm..." << std::endl;
        
        // Make a copy of the graph to work with
        std::unordered_map<int, std::vector<int>> currentGraph = graph.getAdjList();
        
        // Track the best community structure found
        std::vector<std::set<int>> bestCommunities;
        double bestModularity = -1.0;
        int edgesRemoved = 0;
        
        // If maxRemovals is not specified, use estimated number
        if (maxRemovals <= 0) {
            maxRemovals = std::min(500, static_cast<int>(graph.getNumEdges() / 10));
        }
        
        std::cout << "Will remove up to " << maxRemovals << " edges" << std::endl;
        
        // Store initial communities
        bestCommunities = findConnectedComponents(currentGraph);
        bestModularity = calculateModularity(bestCommunities);
        
        // Main loop - remove edges until desired number of communities or max removals
        while (edgesRemoved < maxRemovals) {
            // Calculate edge betweenness
            auto betweenness = calculateEdgeBetweenness();
            
            // Find edge with highest betweenness
            std::pair<int, int> maxEdge = {-1, -1};
            double maxBetweenness = -1.0;
            
            for (const auto& pair : betweenness) {
                if (pair.second > maxBetweenness) {
                    maxBetweenness = pair.second;
                    maxEdge = pair.first;
                }
            }
            
            if (maxEdge.first == -1 || maxEdge.second == -1) {
                std::cout << "No more edges to remove" << std::endl;
                break;
            }
            
            // Remove the edge with highest betweenness
            auto it = std::find(currentGraph[maxEdge.first].begin(), 
                                currentGraph[maxEdge.first].end(), 
                                maxEdge.second);
            if (it != currentGraph[maxEdge.first].end()) {
                currentGraph[maxEdge.first].erase(it);
            }
            
            it = std::find(currentGraph[maxEdge.second].begin(), 
                           currentGraph[maxEdge.second].end(), 
                           maxEdge.first);
            if (it != currentGraph[maxEdge.second].end()) {
                currentGraph[maxEdge.second].erase(it);
            }
            
            edgesRemoved++;
            
            // Check for new communities
            auto communities = findConnectedComponents(currentGraph);
            double modularity = calculateModularity(communities);
            
            // Update best communities if modularity improved
            if (modularity > bestModularity) {
                bestModularity = modularity;
                bestCommunities = communities;
                
                std::cout << "Edge removed: " << maxEdge.first << "-" << maxEdge.second 
                        << ", Edges removed: " << edgesRemoved
                        << ", Communities: " << communities.size() 
                        << ", Modularity: " << modularity << std::endl;
            }
            
            // Optional: Stop if we have many communities or modularity is decreasing
            if (communities.size() > std::sqrt(graph.getNumNodes())) {
                std::cout << "Reached target number of communities" << std::endl;
                break;
            }
            
            // Log progress occasionally
            if (edgesRemoved % 10 == 0) {
                std::cout << "Edges removed: " << edgesRemoved << ", Current communities: " 
                        << communities.size() << ", Best modularity: " << bestModularity << std::endl;
            }
        }
        
        // Convert the best communities to the required format
        std::unordered_map<int, std::set<int>> result;
        for (size_t i = 0; i < bestCommunities.size(); i++) {
            result[i] = bestCommunities[i];
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        std::cout << "\nGirvan-Newman completed in " << elapsed.count() << " seconds" << std::endl;
        std::cout << "Found " << bestCommunities.size() << " communities with modularity " 
                << bestModularity << std::endl;
        
        return result;
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


Graph loadSoftwareGraph() {
    Graph graph;
    std::string filename = "datasets/metabolic/celegans_metabolic.gml";
    
    graph.loadSoftwareFromFileGML(filename);
    return graph;
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
    
    if (argc > 1) num_ants = std::stoi(argv[1]);
    if (argc > 2) num_iterations = std::stoi(argv[2]);
    if (argc > 3) alpha = std::stod(argv[3]);
    if (argc > 4) beta = std::stod(argv[4]);
    if (argc > 5) rho = std::stod(argv[5]);
    if (argc > 6) q0 = std::stod(argv[6]);
    
    // Load graph
    std::cout << "Loading graph..." << std::endl;
    Graph graph = loadSoftwareGraph();

    std::cout<<"Graph loaded--\n";
    
    // Create ACO algorithm
    NodeCommunityACO aco(graph, num_ants, num_iterations, alpha, beta, rho, q0);
    
    // Get start time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run ACO
    std::unordered_map<int, std::set<int>> communities = aco.run();
    
    // Get end time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "\nNode-Community ACO completed in " 
              << elapsed.count() << " seconds" << std::endl;
    
    // Print communities
    printCommunities(communities);
    
    // Save results
    saveGraphWithCommunities(graph, communities, "sequential_communities.txt");

    // In main, after loading the graph but before running your ACO algorithm
    std::cout << "\nRunning Girvan-Newman for comparison..." << std::endl;
    GirvanNewman gn(graph);

    // Get start time for Girvan-Newman
    auto gn_start_time = std::chrono::high_resolution_clock::now();

    // Find communities using Girvan-Newman
    std::unordered_map<int, std::set<int>> gn_communities = gn.findCommunities();

    // Get end time
    auto gn_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gn_elapsed = gn_end_time - gn_start_time;

    std::cout << "\nGirvan-Newman completed in " 
            << gn_elapsed.count() << " seconds" << std::endl;

    // Print communities
    std::cout << "\nGirvan-Newman communities:" << std::endl;
    printCommunities(gn_communities);

    // Save Girvan-Newman results
    saveGraphWithCommunities(graph, gn_communities, "girvan_newman_communities.txt");
    
    return 0;
}