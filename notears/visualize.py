import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
G = nx.DiGraph()

# raw_data = pd.read_csv('data.csv')
# labels = raw_data.columns.values.tolist()
# print(raw_data.head)
adjacent = pd.read_csv('W_est_metabric.csv', header=None)
# adjacent = pd.read_csv('W_est_metabric_classification.csv', header=None)
# adjacent = pd.read_csv('W_est_boston.csv', header=None)
# adjacent = pd.read_csv('W_est_boston_original.csv', header=None)
# adjacent = pd.read_csv('W_est_metabric_original.csv', header=None)
# print(adjacent.head(5))
# exit(0)
adjacent.reset_index(drop=True, inplace=True)

rows = adjacent.shape[0]
columns = adjacent.shape[1]
adjacent = adjacent.to_numpy()

for i in range(rows):
    for j in range(columns):
        if adjacent[i][j] != 0:
            G.add_edge(i, j)

pos = nx.spring_layout(G, k=10)
nx.draw(G, pos=pos, with_labels=True)
plt.show()
