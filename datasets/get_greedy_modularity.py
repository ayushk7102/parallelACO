import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
import re
import numpy as np

def load_dblp_graph(filename):
    """Load the DBLP graph from a file."""
    G = nx.Graph()
    
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Skip comments and empty lines
                if line.startswith('#') or not line.strip():
                    continue
                
                # Try to parse the line as an edge
                try:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        from_node, to_node = int(parts[0]), int(parts[1])
                        G.add_edge(from_node, to_node)
                except ValueError:
                    # Skip header or malformed lines
                    continue
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None
    
    print(f"Graph loaded successfully!")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G


def load_custom_gml(filename):
    """Load a graph from a custom GML-like file."""
    G = nx.Graph()
    try:
        with open(filename, 'r') as file:
            content = file.read()
            
        # Extract all edges using regex pattern matching
        # This is a simple approach and might need adjustment
        node_pattern = r'node\s*\[\s*id\s+(\d+)'
        nodes = re.findall(node_pattern, content)
        
        # Find all edges - assuming they're defined with source and target attributes
        edge_pattern = r'edge\s*\[\s*source\s+(\d+)\s+target\s+(\d+)'
        edges = re.findall(edge_pattern, content)
        
        # Add nodes and edges to the graph
        for node_id in nodes:
            G.add_node(int(node_id))
            
        for source, target in edges:
            G.add_edge(int(source), int(target))
        
        # If no edges were found with the regex, try alternative parsing
        if G.number_of_edges() == 0:
            print("No edges found with standard pattern, trying alternative parsing...")
            
            # Reset graph since the first attempt didn't work
            G = nx.Graph()
            
            # Try to find lines with explicit node connections
            with open(filename, 'r') as file:
                in_edge_section = False
                source_node = None
                
                for line in file:
                    line = line.strip()
                    
                    # Check if we're starting a new node definition
                    if "node [" in line:
                        in_edge_section = False
                        node_id = None
                        continue
                        
                    # Extract node ID if we're inside a node definition
                    if not in_edge_section and "id" in line and not "edge" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                node_id = int(parts[1].strip('"'))
                                G.add_node(node_id)
                            except ValueError:
                                pass
                            
                    # Check if we're starting an edge definition
                    if "edge [" in line:
                        in_edge_section = True
                        source_node = None
                        target_node = None
                        continue
                        
                    # Extract source and target if we're inside an edge definition
                    if in_edge_section and "source" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                source_node = int(parts[1].strip('"'))
                            except ValueError:
                                pass
                            
                    if in_edge_section and "target" in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                target_node = int(parts[1].strip('"'))
                                if source_node is not None and target_node is not None:
                                    G.add_edge(source_node, target_node)
                            except ValueError:
                                pass
        
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None
    except Exception as e:
        print(f"Error loading custom GML file: {str(e)}")
        return None
    
    print(f"Custom GML loaded successfully!")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G

def detect_communities(G):
    """Detect communities using Louvain method."""
    print("Detecting communities...")
    
    # Count the number of communities
    communities = community.greedy_modularity_communities(G)
    modularity = (community.modularity(G, communities))

    print(f"Found {len(communities)} communities")
    # print(communities)
    print("Modularity : " , modularity)
    partition = {}
    for i, comm in enumerate(communities):
        for node in comm:
            partition[node] = i
    return partition, modularity

def visualize_communities(G, partition, max_nodes=100):
    """Visualize communities (limited to max_nodes for better visualization)."""
    # if G.number_of_nodes() > max_nodes:
    #     print(f"Graph is too large, visualizing a subset of {max_nodes} nodes")
    #     G = nx.Graph(G.subgraph(list(G.nodes())[:max_nodes]))
    #     partition = {k: v for k, v in partition.items() if k in G.nodes()}
    
    # Set up the plot
    plt.figure(figsize=(10, 8))
    
    # Draw the graph with communities highlighted by color
    pos = nx.spring_layout(G, seed=42)
    cmap = plt.cm.rainbow
    
    nx.draw_networkx_nodes(G, pos, node_size=40, 
                          node_color=list(partition.values()), 
                          cmap=cmap)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    plt.title("Community Structure of Graph using Greedy NetworkX")
    plt.axis('off')
    plt.savefig("datasets/metabolic/GreedyCommunities.png")
    plt.show()

def main():
    # Load the graph
    # filename = "datasets/DBLP/com-dblp.ungraph.txt"
    filename = "datasets/metabolic/celegans_metabolic.gml"
    G = load_custom_gml(filename)
    
    if G is None:
        return
    
    # Detect communities
    partition, modularity = detect_communities(G)
    visualize_communities(G, partition)


if __name__ == "__main__":
    main()