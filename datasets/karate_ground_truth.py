import networkx as nx

zkc = nx.karate_club_graph()
gt_membership = [(v, zkc.nodes[v]['club']) for v in zkc.nodes()]
print(gt_membership)
