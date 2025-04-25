import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def read_network_data(filename):
    """Read the network data from the file."""
    adjacency_list = {}
    communities = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find where the community information starts
    community_start = 0
    for i, line in enumerate(lines):
        if "# Community_ID Size Nodes" in line:
            community_start = i
            break
    
    # Parse adjacency list
    adjacency_section = lines[1:community_start]
    for line in adjacency_section:
        if "->" in line:
            parts = line.strip().split("->")
            if len(parts) == 2:
                node = int(parts[0].strip())
                neighbors = [int(n) for n in parts[1].strip().split()]
                adjacency_list[node] = neighbors
    
    # Parse community information
    community_section = lines[community_start+1:]
    for line in community_section:
        parts = line.strip().split()
        if len(parts) >= 3:  # Ensure there's at least community_id, size, and one node
            community_id = int(parts[0])
            size = int(parts[1])
            nodes = [int(n) for n in parts[2:]]
            communities[community_id] = nodes
    
    return adjacency_list, communities

def create_graph(adjacency_list):
    """Create a NetworkX graph from the adjacency list."""
    G = nx.Graph()
    
    # Add nodes and edges from the adjacency list
    for node, neighbors in adjacency_list.items():
        G.add_node(node)
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    return G

def visualize_network(G, communities):
    """Visualize the network with communities."""
    # Create a dictionary mapping nodes to their community
    node_to_community = {}
    for community_id, nodes in communities.items():
        for node in nodes:
            node_to_community[node] = community_id
    
    # Generate colors for communities
    num_communities = len(communities)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_communities))
    community_colors = {comm_id: colors[i] for i, comm_id in enumerate(communities.keys())}
    
    # Assign colors to nodes based on their community
    node_colors = [community_colors[node_to_community[node]] if node in node_to_community else 'gray' 
                  for node in G.nodes()]
    
    # Create a larger figure for better visibility
    plt.figure(figsize=(12, 10))
    
    # Use spring layout for node positioning
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the network
    nx.draw(G, pos, 
            node_color=node_colors, 
            node_size=80, 
            edge_color='gray', 
            width=0.5, 
            alpha=0.8,
            with_labels=True,
            font_size=8)
    
    # Create a legend for communities
    for i, comm_id in enumerate(communities.keys()):
        plt.plot([], [], 'o', color=community_colors[comm_id], label=f'Community {comm_id}')
    
    plt.legend(loc='upper right', ncol=2)
    plt.title("Network Communities Visualization")
    plt.axis('off')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig("network_communities.png", dpi=300)
    plt.show()
    
    # Print some network statistics
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Number of communities: {len(communities)}")
    
    # Find the largest community
    largest_comm = max(communities.items(), key=lambda x: len(x[1]))
    print(f"Largest community: {largest_comm[0]} with {len(largest_comm[1])} nodes")

def main():
    # Filename for the network data
    filename = "parallel_communities.txt"
    
    # Read the network data
    adjacency_list, communities = read_network_data(filename)
    
    # Create the graph
    G = create_graph(adjacency_list)
    
    # Visualize the network
    visualize_network(G, communities)

if __name__ == "__main__":
    main()