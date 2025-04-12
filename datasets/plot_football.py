import urllib.request
import io
import zipfile
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community
import numpy as np

url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"
sock = urllib.request.urlopen(url)  # open URL
s = io.BytesIO(sock.read())  # read into BytesIO "file"
sock.close()
zf = zipfile.ZipFile(s)  # zipfile object
txt = zf.read("football.txt").decode()  # read info file
gml = zf.read("football.gml").decode()  # read gml data

gml = gml.split("\n")[1:]
G = nx.parse_gml(gml)  # parse gml data
print(txt)

# Detect communities using Louvain method
communities = community.greedy_modularity_communities(G)
print("Modularity = ", community.modularity(G, communities))
# Create a dictionary mapping nodes to their community index
community_map = {}
for i, comm in enumerate(communities):
    for node in comm:
        community_map[node] = i

# Get a list of colors for each community
import matplotlib.cm as cm
num_communities = len(communities)
print("Ground truth num_communities: ", num_communities)
colors = cm.rainbow(np.linspace(0, 1, num_communities))

# Create a list of node colors based on their community
node_colors = [colors[community_map[node]] for node in G.nodes()]

# Print degree for each team - number of games
# for n, d in G.degree():
#     print(f"{n:20} {d:2}")

# Draw the graph with community colors
options = {
    "node_size": 50,
    "linewidths": 0,
    "width": 0.1,
    "node_color": node_colors
}
pos = nx.spring_layout(G, seed=1969)  # Seed for reproducible layout

# for i, comm in enumerate(communities):
#     print(f"Community {i}: {sorted(list(comm))}")

nx.draw(G, pos, **options)
plt.show()