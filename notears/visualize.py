import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
G = nx.DiGraph()

# raw_data = pd.read_csv('data.csv')
# labels = raw_data.columns.values.tolist()
# print(raw_data.head)
# adjacent = pd.read_csv('W_est_metabric.csv', header=None)
adjacent = pd.read_csv('W_est_metabric_classification.csv', header=None)
# adjacent = pd.read_csv('W_est_boston.csv', header=None)
# adjacent = pd.read_csv('W_est_boston_original.csv', header=None)
# adjacent = pd.read_csv('W_est_metabric_original.csv', header=None)
# print(adjacent.head(5))
# exit(0)
adjacent.reset_index(drop=True, inplace=True)

# print(adjacent)

rows = adjacent.shape[0]
columns = adjacent.shape[1]
# print(rows,columns)

adjacent = adjacent.to_numpy()


# print(adjacent.shape)

# exit(0)

for i in range(rows):
    for j in range(columns):
        if adjacent[i][j] != 0:
            G.add_edge(i, j)

# ans=[]
# for i in range(rows):
#     if adjacent[i][13]!=0:
#         ans.append(i)

# print('ans', ans)

#CHAS, RM, DIS, PT-RATIO

# G.add_edge(50,60)

# nodes = [0, 1, 21, 26]
# for i in nodes:
#     print(labels[i+1])
pos = nx.spring_layout(G, k=3)
nx.draw(G, pos=pos, with_labels=True)
plt.show()
