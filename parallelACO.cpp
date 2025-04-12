#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

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

    // Additional methods for graph analysis can be added here
};


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

int main() {
    Graph graph;
    
    // graph = loadDBLPGraph();
    graph = loadFootballGraph();
    graph.printGraph();
    
    return 0;
}