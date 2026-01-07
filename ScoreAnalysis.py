import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from GradientDecent import stochastic_gradient_descent


class ScoreAnalysis:
    def __init__(self, scores):
        self.scores = scores

    def statistical_summary(self):
        print("Statistical Summary:")
        print(f"Minimum Score: {np.min(self.scores)}")
        print(f"Maximum Score: {np.max(self.scores)}")
        print(f"Mean Score: {np.mean(self.scores)}")
        print(f"Median Score: {np.median(self.scores)}")
        print(f"Standard Deviation: {np.std(self.scores)}")

    def plot_histogram(self):
        plt.hist(self.scores, bins=30, edgecolor='black')
        plt.title("Histogram of Predicted Fusion Scores")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.show()

    def plot_boxplot(self):
        plt.boxplot(self.scores, vert=False)
        plt.title("Box Plot of Predicted Fusion Scores")
        plt.xlabel("Score")
        plt.show()




G = nx.Graph()

with open('out.gene_fusion', 'r') as file:
    for line in file:
        if line.startswith('%'):  # Skip comments
            continue
        node1, node2 = map(int, line.split())
        G.add_edge(node1, node2)

Y_matrix = nx.adjacency_matrix(G).todense()
np.save('Y_matrix.npy', Y_matrix)

rank = 5

Z = stochastic_gradient_descent(G, rank)

num_nodes = G.number_of_nodes()
Y_hat = np.zeros((num_nodes, num_nodes))

# dot products of Z features
for i in range(num_nodes):
    for j in range(num_nodes):
        Y_hat[i, j] = np.dot(Z[i, :], Z[j, :])

np.save('Y_hat_matrix.npy', Y_hat)

Y_hat = np.load('Y_hat_matrix.npy')
Y = np.load('Y_matrix.npy')

upper_triangular_indices = np.triu_indices_from(Y_hat, k=1)
observed_edges_scores = []
for i, j in G.edges():
    observed_edges_scores.append(Y_hat[i, j])
observed_edges_scores = np.array(observed_edges_scores)

upper_triangular_indices = np.triu_indices_from(Y_hat, k=1)
unobserved_mask = Y[upper_triangular_indices] == 0
unobserved_edges_scores = Y_hat[upper_triangular_indices][unobserved_mask]

print("Observed edges scores:")
print("  mean   :", np.mean(observed_edges_scores))
print("  median :", np.median(observed_edges_scores))
print("  std    :", np.std(observed_edges_scores))

print("\nUnobserved edges scores:")
print("  mean   :", np.mean(unobserved_edges_scores))
print("  median :", np.median(unobserved_edges_scores))
print("  std    :", np.std(unobserved_edges_scores))

plt.figure(figsize=(10, 5))
plt.hist(observed_edges_scores, bins=30, alpha=0.6, label="Observed edges")
plt.hist(unobserved_edges_scores, bins=30, alpha=0.6, label="Unobserved edges")
plt.xlabel("Z_i · Z_j score")
plt.ylabel("Frequency")
plt.title("Observed vs Unobserved Edge Score Distribution")
plt.legend()
plt.show()