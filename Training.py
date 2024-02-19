import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from GradientDecent import stochastic_gradient_descent


class GeneFusionPredictor:
    def __init__(self, rank=2, lambda_reg=0.1, max_iter=1000, epsilon=1e-3, clip_value=1e5, node_map=None):
        """
        Initialize the predictor with the necessary hyperparameters and node map.
        """
        self.rank = rank
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.Z = None
        self.threshold = None
        self.node_map = node_map  # Store node_map



    def fit(self, graph, positive_examples, negative_examples):
        """
        Train the model using stochastic gradient descent on the provided graph.
        """
        self.Z = stochastic_gradient_descent(
            graph,
            self.rank,
            self.epsilon,
            self.max_iter,
            self.lambda_reg,
            self.clip_value
        )

        # Calculate similarities for known fusions and non-fusions
        positive_similarities = [np.dot(self.Z[i], self.Z[j]) for (i, j) in positive_examples if
                                 i in self.Z and j in self.Z]
        negative_similarities = [np.dot(self.Z[i], self.Z[j]) for (i, j) in negative_examples if
                                 i in self.Z and j in self.Z]

        # Handle empty lists
        if positive_similarities:
            positive_mean = np.mean(positive_similarities)
        else:
            positive_mean = 0

        if negative_similarities:
            negative_mean = np.mean(negative_similarities)
        else:
            negative_mean = 0

        # Determine threshold
        self.threshold = (positive_mean + negative_mean) / 2


    def predict(self, i, j):
        """
        Predict whether a pair of genes (i, j) will fuse.
        """
        if self.Z is None:
            raise ValueError("Model has not been trained yet. Call the fit method first.")

        if i >= len(self.Z) or j >= len(self.Z):
            # Handle invalid indices
            return False

        similarity = np.dot(self.Z[i], self.Z[j])
        return similarity > self.threshold


def prepare_data(graph, edges, node_map):
    """
    Prepares data for training the model.
    """
    X = graph  # In this case, the feature is the graph itself
    y = []

    # Iterate over all possible node pairs
    for node1 in graph.nodes():
        for node2 in graph.nodes():
            if node1 != node2:
                # Check if an edge exists (fusion case)
                if graph.has_edge(node1, node2):
                    y.append(1)  # Fusion
                else:
                    y.append(0)  # No fusion

    return X, y


def remap_nodes(edges):
    """
    Remap the node indices in the edges to a contiguous range starting from 0.
    """
    unique_nodes = set([n for edge in edges for n in edge])
    node_map = {node: i for i, node in enumerate(unique_nodes)}
    remapped_edges = [(node_map[i], node_map[j]) for i, j in edges]
    return remapped_edges, node_map



def split_data(edges, train_ratio=0.7, validation_ratio=0.15):
    """
    Function to split data into training and testing sets
    """
    random.shuffle(edges)
    train_index = int(len(edges) * train_ratio)
    validation_index = int(len(edges) * (train_ratio + validation_ratio))
    return edges[:train_index], edges[train_index:validation_index], edges[validation_index:]


def evaluate_model(model, testing_edges, graph, node_map):
    """
    Function to calculate performance metrics
    """
    true_labels = []
    predictions = []

    for i, j in testing_edges:
        if i in node_map and j in node_map:
            true_labels.append(graph.has_edge(i, j))
            predictions.append(model.predict(node_map[i], node_map[j]))
        else:
            # Handle edges with nodes not in node_map
            true_labels.append(False)
            predictions.append(False)

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    return accuracy, precision, recall, f1


# Read and process gene fusion data
file_path = 'out.gene_fusion'
all_edges = []
with open(file_path, 'r') as file:
    for line in file:
        if line.startswith('%'):
            continue
        node1, node2 = map(int, line.strip().split())
        all_edges.append((node1, node2))

# Remap node indices and create the graph
remapped_edges, node_map = remap_nodes(all_edges)
G = nx.Graph()
G.add_edges_from(remapped_edges)

# Ensure you pass the correct node_map
model = GeneFusionPredictor(rank=2, lambda_reg=0.1, node_map=node_map)

X, y = prepare_data(G, all_edges, node_map)
best_score = 0
best_params = None

# Split dataset into training, validation, and testing sets
training_edges, validation_edges, testing_edges = split_data(remapped_edges)

# Hyperparameter tuning loop
for rank in [2, 5, 10]:
    for lambda_reg in [0.01, 0.1, 1]:
        model = GeneFusionPredictor(rank=rank, lambda_reg=lambda_reg, node_map=node_map)

        # Train using training data
        train_graph = nx.Graph()
        train_graph.add_nodes_from(range(len(node_map)))
        train_graph.add_edges_from(training_edges)
        model.fit(train_graph, training_edges, [])

        # Evaluate using validation data
        accuracy, precision, recall, f1 = evaluate_model(model, validation_edges, G, node_map)

        # Update best parameters if better
        if accuracy > best_score:
            best_score = accuracy
            best_params = {'rank': rank, 'lambda_reg': lambda_reg}

# Train final model using training and validation data
final_model = GeneFusionPredictor(**best_params, node_map=node_map)
final_graph = nx.Graph()
final_graph.add_nodes_from(range(len(node_map)))
final_graph.add_edges_from(training_edges + validation_edges)
final_model.fit(final_graph, training_edges + validation_edges, [])

# Final evaluation on the test set
final_accuracy, final_precision, final_recall, final_f1 = evaluate_model(final_model, testing_edges, G, node_map)
print(f"Final model evaluation:\nAccuracy: {final_accuracy}\nPrecision: {final_precision}\nRecall: {final_recall}\nF1 Score: {final_f1}")







"""
Plot fraph network
"""
G = nx.convert_node_labels_to_integers(G)

# Parameters
r = 2
epsilon = 0.001
lambda_reg = 0.01

Z = stochastic_gradient_descent(G, r, epsilon, lambda_reg)

# Visualization
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
