import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from GradientDecent import stochastic_gradient_descent


class Testing:
    def __init__(self, rank, lambda_reg, epsilon):
        self.rank = rank
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon
        self.graphs = {}
        self.adjacency_matrices = {}

    def add_graph(self, graph_name, G):
        """
        Add a graph and its corresponding adjacency matrix to the class.
        """
        self.graphs[graph_name] = G
        self.adjacency_matrices[graph_name] = self.create_adjacency_matrix(G)


    @staticmethod
    def create_adjacency_matrix(G):
        """
        Create an adjacency matrix representation of the graph.
        """
        return nx.to_numpy_array(G)


    @staticmethod
    def display_graph(G):
        """
        Plots the graph with nodes and edges
        """
        plt.figure()
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        plt.show()


    def test_sgd_on_graph(self, graph_name):
        """
        Applies the stochastic gradient descent algorithm to a specified graph.
        """
        G = self.graphs.get(graph_name)
        if G is None:
            print(f"No graph found with the name '{graph_name}'.")
            return None

        Z = stochastic_gradient_descent(G, self.rank, epsilon=self.epsilon, lambda_reg=self.lambda_reg, max_iter=1000)
        if Z is None or len(Z) == 0:
            print("stochastic_gradient_descent returned an empty result.")
            return None

        return Z


    def run_multiple_tests(self, graph_name, n_runs=5):
        """
        Run test_sgd_on_graph multiple times to check the consistency of the results.
        """
        consistency_results = []
        for _ in range(n_runs):
            Z = self.test_sgd_on_graph(graph_name)
            consistency_results.append(Z)
        return consistency_results


    def visualize_Z(self, Z, graph_name):
        """
        Create a visualization of the latent space representation Z matrix.
        """
        plt.figure()
        for i, point in enumerate(Z):
            plt.scatter(point[0], point[1], label=f'Node {i}')
        plt.title(f'2D Visualization of Z for {graph_name}')
        plt.legend()
        plt.show()


    def reconstruct_adjacency_matrix(self, Z):
        """
        Reconstruct the adjacency matrix from the low-rank approximation.
        """
        return np.dot(Z, Z.T)


    def apply_threshold(self, reconstructed_Y, threshold=0.5):
        """
        Apply a threshold to the reconstructed adjacency matrix.
        """
        return (reconstructed_Y >= threshold).astype(int)


    def compare_matrices(self, original_Y, reconstructed_Y, threshold=0.5):
        """
        Compare the original and reconstructed adjacency matrices.
        """
        thresholded_Y = self.apply_threshold(reconstructed_Y, threshold)
        difference = np.linalg.norm(original_Y - thresholded_Y, 'fro')
        similarity = np.mean(original_Y == thresholded_Y)
        return difference, similarity


    def find_optimal_threshold(self, Y, reconstructed_Y, num_thresholds=50):
        """
        Find the optimal threshold that maximizes the similarity between the original and reconstructed matrices.
        """
        best_similarity = 0
        best_threshold = 0
        for threshold in np.linspace(0, 1, num_thresholds):
            thresholded_Y = self.apply_threshold(reconstructed_Y, threshold)
            similarity = np.mean(Y == thresholded_Y)
            if similarity > best_similarity:
                best_similarity = similarity
                best_threshold = threshold
        return best_threshold, best_similarity




tester = Testing(rank=2, lambda_reg=0.1, epsilon=1e-3)

# Add small graphs for initial testing
small_graphs = {
    'Graph 1': nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),
    'Graph 2': nx.Graph([(0, 1), (1, 2), (2, 0)]),
    'Graph 3': nx.Graph([(0, 1), (1, 2), (2, 3)])
}

for name, G in small_graphs.items():
    tester.add_graph(name, G)

# Testing with small graphs
for graph_name in small_graphs.keys():
    print(f"\nTesting {graph_name} with Parameters: Rank={tester.rank}, Lambda={tester.lambda_reg}, Epsilon={tester.epsilon}")

    # Apply SGD on the graph
    Z_matrix = tester.test_sgd_on_graph(graph_name)

    if Z_matrix is None or len(Z_matrix) == 0:
        print(f"No Z matrix to visualize or test further for {graph_name}.")
        continue

    consistency_results = tester.run_multiple_tests(graph_name, n_runs=5)
    for i, Z in enumerate(consistency_results, 1):
        print(f"Run {i} for {graph_name}:")
        print(Z)

    # Reconstruct adjacency matrix from the Z matrix
    reconstructed_Y = tester.reconstruct_adjacency_matrix(Z_matrix)

    # Find optimal threshold for the reconstructed matrix
    Y = tester.adjacency_matrices[graph_name]
    best_threshold, best_similarity = tester.find_optimal_threshold(Y, reconstructed_Y)
    print(f"Best Threshold for {graph_name}: {best_threshold}")
    print(f"Best Similarity for {graph_name}: {best_similarity:.2%}")

    # Display the original graph
    tester.display_graph(tester.graphs[graph_name])



"""
Test in dataset
"""
print("\nTest on dataset Gene Fusion")
G_gene_fusion = nx.read_edgelist('out.gene_fusion', nodetype=str, data=False)
tester.add_graph('Gene Fusion Graph', G_gene_fusion)

# Testing the 'Gene Fusion Graph' with different parameter combinations
ranks = [2, 3, 4, 5]
lambda_regs = [0.01, 0.05, 0.1, 0.5]
epsilons = [1e-3, 1e-4, 1e-5]

best_performance = {'rank': None, 'lambda_reg': None, 'epsilon': None, 'similarity': 0}

for rank in ranks:
    for lambda_reg in lambda_regs:
        for epsilon in epsilons:
            tester.rank = rank
            tester.lambda_reg = lambda_reg
            tester.epsilon = epsilon

            Z_matrix = tester.test_sgd_on_graph('Gene Fusion Graph')
            if Z_matrix is None or len(Z_matrix) == 0:
                continue

            reconstructed_Y = tester.reconstruct_adjacency_matrix(Z_matrix)
            Y = tester.adjacency_matrices['Gene Fusion Graph']
            _, similarity = tester.find_optimal_threshold(Y, reconstructed_Y)

            # Update best parameters if current similarity is better
            if similarity > best_performance['similarity']:
                best_performance = {
                    'rank': rank,
                    'lambda_reg': lambda_reg,
                    'epsilon': epsilon,
                    'similarity': similarity
                }

print("Best Performing Parameters:", best_performance)

# Re-initialize tester with the best parameters for the 'Gene Fusion Graph'
best_rank = best_performance['rank']
best_lambda_reg = best_performance['lambda_reg']
best_epsilon = best_performance['epsilon']
tester = Testing(best_rank, best_lambda_reg, best_epsilon)
tester.add_graph('Gene Fusion Graph', G_gene_fusion)

# Output consistency results for the 'Gene Fusion Graph'
consistency_results = tester.run_multiple_tests('Gene Fusion Graph', n_runs=5)
for i, Z in enumerate(consistency_results, 1):
    print(f"Run {i} for Gene Fusion Graph:")
    print(Z)

# Test algorithm on the 'Gene Fusion Graph' with the best parameters
Z_matrix = tester.test_sgd_on_graph('Gene Fusion Graph')
if Z_matrix is not None and len(Z_matrix) > 0:
    # Reconstruct adjacency matrix
    reconstructed_Y = tester.reconstruct_adjacency_matrix(Z_matrix)
    original_Y = tester.adjacency_matrices['Gene Fusion Graph']

    # Compare reconstructed and original adjacency matrices
    difference, similarity = tester.compare_matrices(original_Y, reconstructed_Y)
    print(f"Difference: {difference}, Similarity: {similarity}")

    # Find optimal threshold
    best_threshold, best_similarity = tester.find_optimal_threshold(original_Y, reconstructed_Y)
    print(f"Best Threshold: {best_threshold}, Best Similarity: {best_similarity}")
else:
    print("No Z matrix was generated.")



