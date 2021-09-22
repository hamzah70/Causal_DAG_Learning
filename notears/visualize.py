import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
G = nx.DiGraph()

raw_data = pd.read_csv('data.csv')
labels = raw_data.columns.values.tolist()
print(raw_data.head)
adjacent = pd.read_csv('W_est.csv', header=None)
print(adjacent.head)
# exit(0)
adjacent.reset_index(drop=True, inplace=True)

# print(adjacent)

rows = adjacent.shape[0]
columns = adjacent.shape[1]
# print(rows,columns)

adjacent = adjacent.to_numpy()


# print(adjacent)

# exit(0)

for i in range(rows):
    for j in range(columns):
        if adjacent[i][j] != 0:
            G.add_edge(i, j)

nodes = [0, 1, 21, 26]
for i in nodes:
    print(labels[i+1])
nx.draw(G, with_labels=True)
plt.show()
