import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community

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

def load_graphml(filename):
    """Load a graph from a GraphML file."""
    try:
        G = nx.read_graphml(filename)
        print(f"GraphML loaded successfully!")
        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        return G
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None
    except Exception as e:
        print(f"Error loading GraphML file: {str(e)}")
        return None

def detect_communities(G):
    """Detect communities using Louvain method."""
    print("Detecting communities...")
    
    # Count the number of communities
    communities = community.greedy_modularity_communities(G)
    modularity = (community.modularity(G, communities))

    print(f"Found {len(communities)} communities")
    
    return communities, modularity

def visualize_communities(G, partition, max_nodes=100):
    """Visualize communities (limited to max_nodes for better visualization)."""
    if G.number_of_nodes() > max_nodes:
        print(f"Graph is too large, visualizing a subset of {max_nodes} nodes")
        G = nx.Graph(G.subgraph(list(G.nodes())[:max_nodes]))
        partition = {k: v for k, v in partition.items() if k in G.nodes()}
    
    # Set up the plot
    plt.figure(figsize=(10, 8))
    
    # Draw the graph with communities highlighted by color
    pos = nx.spring_layout(G, seed=42)
    cmap = plt.cm.rainbow
    
    nx.draw_networkx_nodes(G, pos, node_size=40, 
                          node_color=list(partition.values()), 
                          cmap=cmap)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    plt.title("Community Structure of DBLP Graph")
    plt.axis('off')
    plt.savefig("dblp_communities.png")
    plt.show()

def main():
    # Load the graph
    # filename = "datasets/DBLP/com-dblp.ungraph.txt"
    filename = "datasets/metabolic/celegans_metabolic.gml"
    G = load_dblp_graph(filename)
    
    if G is None:
        return
    
    # Detect communities
    communities, modularity = detect_communities(G)
        

if __name__ == "__main__":
    main()