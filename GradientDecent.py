import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


def stochastic_gradient_descent(graph, rank, epsilon=1e-3, max_iter=1000, lambda_reg=0.1, clip_value=1e5):
    """
    Performs the sequential stochastic gradient descent algorithm on a given graph.
    The algorithm stops when the Frobenius norm of the difference in Z between
    two consecutive iterations is less than epsilon or when max_iter is reached.
    """
    max_iter = int(max_iter)
    # Create a mapping from node labels to indices
    node_indices = {node: idx for idx, node in enumerate(graph.nodes())}
    # Initialize Z with small random values to prevent overflow
    Z = np.random.rand(graph.number_of_nodes(), rank) * 0.01

    # Repeat until convergence or maximum iterations
    for it in range(max_iter):
        Z_prev = Z.copy()

        # Update Zi for each node i
        for i in graph.nodes():
            i_idx = node_indices[i]  # Convert node label to index
            Zi = Z_prev[i_idx, :]
            grad_sum = np.zeros(rank)

            for j in graph.neighbors(i):
                j_idx = node_indices[j]  # Convert node label to index
                Zj = Z_prev[j_idx, :]
                Yij = graph[i][j]['weight'] if 'weight' in graph[i][j] else 1
                grad = (Yij - np.dot(Zi, Zj)) * Zj
                grad_sum += grad

            # Clip the gradient to prevent overflow (this is essential to prevent overflow)
            grad_sum = np.clip(grad_sum, -clip_value, clip_value)

            # Learning rate: decreases with iteration
            eta = 1.0 / np.sqrt(it + 1)

            # Update rule for Zi
            Z[i_idx, :] = Zi + eta * (grad_sum - lambda_reg * Zi)

        # Check for convergence
        norm_diff = np.linalg.norm(Z - Z_prev, 'fro')
        if norm_diff < epsilon:
            break

    return Z






