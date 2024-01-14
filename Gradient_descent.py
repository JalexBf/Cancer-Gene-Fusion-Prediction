import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def stochastic_gradient_descent(Y, r, epsilon, lambda_reg):
    n = Y.shape[0]  # number of nodes
    Z = np.random.rand(n, r)  # αρχικοποιηση Ζ με τυχαιους
    t = 1  # Time step

    while True:
        Z_prime = Z.copy()
        for i, j in G.edges():
            eta = 1 / np.sqrt(t)
            t += 1
            gradient = (np.dot(Z[i], Z[j].T) - Y[i, j]) * Z[j] + lambda_reg * Z[i]
            Z[i] -= eta * gradient  # Update Z[i]
            #Z[j] -= eta * gradient  # Update Z[j] as well

        if np.linalg.norm(Z - Z_prime, 'fro') < epsilon:
            break

    return Z


G = nx.read_edgelist('out.gene_fusion', create_using=nx.Graph(), data=False)

# Relabel nodes to ensure they are consecutive integers starting from 0
G = nx.convert_node_labels_to_integers(G)

# Convert graph to an adjacency matrix
Y = nx.to_numpy_array(G)

r = 2
epsilon = 0.001
lambda_reg = 0.01
Z = stochastic_gradient_descent(Y, r, epsilon, lambda_reg)

print(Z)

if Z.shape[1] == 2:
    plt.figure(figsize=(12, 8))
    plt.scatter(Z[:, 0], Z[:, 1], s=50, c='blue')

    # Draw lines between connected nodes
    for i, j in G.edges():
        plt.plot([Z[i, 0], Z[j, 0]], [Z[i, 1], Z[j, 1]], 'grey', alpha=0.5)

    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.title('Node Embeddings with Connections')
    plt.show()

