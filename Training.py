import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from GradientDecent import stochastic_gradient_descent


random.seed(0)
np.random.seed(0)


def canon(u, v):
    return (u, v) if u < v else (v, u)


# Sample non edges using oracle_graph to avoid sampling true edges
def sample_negative_edges(oracle_graph, num_samples, forbidden_edges=None, seed=0):
  
    rng = np.random.default_rng(seed)
    nodes = np.array(list(oracle_graph.nodes()), dtype=int)
    forbidden_edges = forbidden_edges or set()

    deg = np.array([oracle_graph.degree(int(n)) for n in nodes], dtype=float)
    deg_sum = float(deg.sum())
    p = deg / deg_sum if deg_sum > 0 else np.ones(len(nodes)) / len(nodes)

    neg = set()
    while len(neg) < num_samples:
        i = int(rng.choice(nodes, p=p))
        j = int(rng.choice(nodes, p=p))
        if i == j:
            continue
        e = canon(i, j)
        if oracle_graph.has_edge(i, j):
            continue
        if e in forbidden_edges:
            continue
        neg.add(e)

    return list(neg)


# threshold + direction (flip) ONLY using validation
def fit_threshold_on_val(Z, val_pos_edges, oracle_graph, num_thresholds=200, n_resamples=10, seed=0):

    val_pos_set = {canon(i, j) for i, j in val_pos_edges}
    pos_scores = np.array([float(np.dot(Z[i], Z[j])) for i, j in val_pos_edges])

    # presample some neg scores for stability
    neg_scores_pool = []
    for r in range(n_resamples):
        neg_edges = sample_negative_edges(
            oracle_graph,
            len(val_pos_edges),
            forbidden_edges=val_pos_set,
            seed=seed + 1000 + r
        )
        neg_scores_pool.append(np.array([float(np.dot(Z[i], Z[j])) for i, j in neg_edges]))
    neg_scores_pool = np.stack(neg_scores_pool, axis=0)

    all_scores = np.concatenate([pos_scores, neg_scores_pool.reshape(-1)])
    ts = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)

    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(pos_scores))])

    best_f1, best_t, best_flip = -1.0, None, None

    for t in ts:
        f1s_gt, f1s_lt = [], []
        for r in range(n_resamples):
            neg_scores = neg_scores_pool[r]
            y_pred_gt = np.concatenate([(pos_scores > t), (neg_scores > t)]).astype(int)
            y_pred_lt = np.concatenate([(pos_scores < t), (neg_scores < t)]).astype(int)
            f1s_gt.append(f1_score(y_true, y_pred_gt, zero_division=0))
            f1s_lt.append(f1_score(y_true, y_pred_lt, zero_division=0))

        f1_gt = float(np.mean(f1s_gt))
        f1_lt = float(np.mean(f1s_lt))

        if f1_gt > best_f1:
            best_f1, best_t, best_flip = f1_gt, float(t), False
        if f1_lt > best_f1:
            best_f1, best_t, best_flip = f1_lt, float(t), True

    return best_f1, best_t, best_flip


def evaluate_thresholded(Z, pos_edges, oracle_graph, threshold, flip, n_resamples=20, seed=0):
    pos_set = {canon(i, j) for i, j in pos_edges}
    pos_scores = np.array([float(np.dot(Z[i], Z[j])) for i, j in pos_edges])

    accs, precs, recs, f1s = [], [], [], []
    for r in range(n_resamples):
        neg_edges = sample_negative_edges(
            oracle_graph,
            len(pos_edges),
            forbidden_edges=pos_set,
            seed=seed + 2000 + r
        )
        neg_scores = np.array([float(np.dot(Z[i], Z[j])) for i, j in neg_edges])

        if flip:
            y_pred = np.concatenate([(pos_scores < threshold), (neg_scores < threshold)]).astype(int)
        else:
            y_pred = np.concatenate([(pos_scores > threshold), (neg_scores > threshold)]).astype(int)

        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

        accs.append(accuracy_score(y_true, y_pred))
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    return float(np.mean(accs)), float(np.mean(precs)), float(np.mean(recs)), float(np.mean(f1s))


def evaluate_ranking(Z, pos_edges, oracle_graph, n_neg_mult=1, n_resamples=20, seed=0):
    pos_set = {canon(i, j) for i, j in pos_edges}
    pos_raw = np.array([float(np.dot(Z[i], Z[j])) for i, j in pos_edges])

    # direction so that higher is more positive
    neg_dir = sample_negative_edges(oracle_graph, len(pos_edges), forbidden_edges=pos_set, seed=seed + 3000)
    neg_raw_dir = np.array([float(np.dot(Z[i], Z[j])) for i, j in neg_dir])
    direction = 1.0 if float(np.mean(pos_raw)) >= float(np.mean(neg_raw_dir)) else -1.0

    # monotonic score in [0,1]
    def score(x):
        x = np.clip(direction * x, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-x))

    aucs, aps = [], []
    for r in range(n_resamples):
        neg_edges = sample_negative_edges(
            oracle_graph,
            n_neg_mult * len(pos_edges),
            forbidden_edges=pos_set,
            seed=seed + 4000 + r
        )
        neg_raw = np.array([float(np.dot(Z[i], Z[j])) for i, j in neg_edges])

        y_true = np.concatenate([np.ones(len(pos_raw)), np.zeros(len(neg_raw))])
        y_score = np.concatenate([score(pos_raw), score(neg_raw)])

        aucs.append(roc_auc_score(y_true, y_score))
        aps.append(average_precision_score(y_true, y_score))

    return float(np.mean(aucs)), float(np.mean(aps)), float(direction)



class GeneFusionPredictor:
    def __init__(
        self,
        rank=20,
        lambda_reg=1e-3,
        max_iter=50,
        epsilon=1e-3,
        clip_value=5.0,
        neg_per_pos=5,
        lr=0.05,
        seed=0,
    ):
        self.rank = rank
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.clip_value = clip_value
        self.neg_per_pos = neg_per_pos
        self.lr = lr
        self.seed = seed
        self.Z = None

    def fit(self, train_graph, oracle_graph):
        self.Z = stochastic_gradient_descent(
            graph=train_graph,
            rank=self.rank,
            epsilon=self.epsilon,
            max_iter=self.max_iter,
            lambda_reg=self.lambda_reg,
            clip_value=self.clip_value,
            neg_per_pos=self.neg_per_pos,
            oracle_graph=oracle_graph,
            lr=self.lr,
            seed=self.seed,
        )
        return self.Z



edges = []
with open("out.gene_fusion") as f:
    for line in f:
        if not line.startswith("%"):
            i, j = map(int, line.split())
            edges.append((i, j))

G = nx.Graph()
G.add_edges_from(edges)
G = nx.convert_node_labels_to_integers(G)

all_edges = list(G.edges())
random.shuffle(all_edges)

n = len(all_edges)
train = all_edges[: int(0.7 * n)]
val = all_edges[int(0.7 * n) : int(0.85 * n)]
test = all_edges[int(0.85 * n) :]

best_val_f1 = -1.0
best = None  # (rank, lambda_reg, threshold, flip)

for rank in [10, 20, 50]:
    for lambda_reg in [1e-4, 1e-3, 1e-2]:
        train_graph = nx.Graph()
        train_graph.add_nodes_from(G.nodes())
        train_graph.add_edges_from(train)

        model = GeneFusionPredictor(rank=rank, lambda_reg=lambda_reg, max_iter=50, seed=0)
        Z = model.fit(train_graph, oracle_graph=G)

        val_f1, t, flip = fit_threshold_on_val(Z, val, G, seed=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best = (rank, lambda_reg, t, flip)

print("best params:", best, "best val f1:", best_val_f1)

rank, lambda_reg, t, flip = best

final_train_graph = nx.Graph()
final_train_graph.add_nodes_from(G.nodes())
final_train_graph.add_edges_from(train)

final_model = GeneFusionPredictor(rank=rank, lambda_reg=lambda_reg, max_iter=50, seed=0)
Z = final_model.fit(final_train_graph, oracle_graph=G)

# re-fit threshold on val using the final trained embeddings
val_f1, t, flip = fit_threshold_on_val(Z, val, G, seed=1)

acc, prec, rec, f1 = evaluate_thresholded(Z, test, G, threshold=t, flip=flip, n_resamples=20, seed=2)
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1: {f1:.3f}")

auc, ap, direction = evaluate_ranking(Z, test, G, n_neg_mult=1, n_resamples=20, seed=3)
print(f"AUC: {auc:.3f}")
print(f"AP:  {ap:.3f}")
print(f"direction: {direction:+.0f} (higher=positive)")