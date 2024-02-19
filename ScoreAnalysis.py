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

# Read file and add edges
with open('out.gene_fusion', 'r') as file:
    for line in file:
        if line.startswith('%'):  # Skip comments
            continue
        node1, node2 = map(int, line.split())
        G.add_edge(node1, node2)

# Generate adjacency matrix
Y_matrix = nx.adjacency_matrix(G).todense()
np.save('Y_matrix.npy', Y_matrix)

rank = 5

# Run stochastic gradient descent
Z = stochastic_gradient_descent(G, rank)

# Initialize Y_hat matrix
num_nodes = G.number_of_nodes()
Y_hat = np.zeros((num_nodes, num_nodes))

# Compute predicted values (dot products of Z features)
for i in range(num_nodes):
    for j in range(num_nodes):
        Y_hat[i, j] = np.dot(Z[i, :], Z[j, :])

np.save('Y_hat_matrix.npy', Y_hat)

Y_hat = np.load('Y_hat_matrix.npy')
Y = np.load('Y_matrix.npy')

# Compute the upper triangular indices
upper_triangular_indices = np.triu_indices_from(Y_hat, k=1)

# Extract unobserved edges scores
unobserved_edges_scores = Y_hat[upper_triangular_indices]

# Create mask for observed edges
observed_mask = Y[upper_triangular_indices] == 0

# Apply mask to filter scores
unobserved_edges_scores = unobserved_edges_scores[observed_mask]

# Analysis and visualization of scores
score_analysis = ScoreAnalysis(unobserved_edges_scores)
score_analysis.statistical_summary()
score_analysis.plot_histogram()
score_analysis.plot_boxplot()