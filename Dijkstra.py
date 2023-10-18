import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Dijkstra's algorithm example
edgelist = pd.read_csv('StreetNet.csv')
print(edgelist.head())
g = nx.Graph()
for i, elrow in edgelist.iterrows():
    g.add_edge(elrow[0], elrow[1], weight=elrow[2])
nx.drawing.nx_pylab.draw_networkx(g)
plt.show()
print('# of edges: {}'.format(g.number_of_edges()))
print('# of nodes: {}'.format(g.number_of_nodes()))
print("\nShortest Path:", nx.shortest_path(g,source=1,target=10, weight='weight')) # uses dijktra's algorithm by default
print("\nShortest Path Length:", nx.shortest_path_length(g,source=1,target=10, weight='weight'))
