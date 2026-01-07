import numpy as np


def sigmoid(x):
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def stochastic_gradient_descent(
    graph,
    rank,
    epsilon=1e-3,
    max_iter=50,        
    lambda_reg=1e-3,
    clip_value=5.0,
    neg_per_pos=5,
    oracle_graph=None,    # only for negative sampling 
    lr=0.05,
    seed=0,
):

    rng = np.random.default_rng(seed)

    nodes = list(graph.nodes())
    node_indices = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)

    Z = rng.normal(0.0, 0.1, size=(n, rank))

    pos_edges = [(node_indices[u], node_indices[v]) for u, v in graph.edges()]
    if not pos_edges:
        return Z

    oracle = oracle_graph if oracle_graph is not None else graph

    # degree proportional negative sampling
    deg = np.array([oracle.degree(node) for node in nodes], dtype=float)
    deg_sum = float(deg.sum())
    p = deg / deg_sum if deg_sum > 0 else np.ones(n) / n

    def sample_negative(i_idx):
        i_label = nodes[i_idx]
        while True:
            k_idx = int(rng.choice(n, p=p))
            if k_idx == i_idx:
                continue
            k_label = nodes[k_idx]
            # reject if its a true edge anywhere in oracle
            if not oracle.has_edge(i_label, k_label):
                return k_idx

    prev_Z = Z.copy()

    for _epoch in range(int(max_iter)):
        rng.shuffle(pos_edges)

        for (i, j) in pos_edges:
            Zi = Z[i]
            Zj = Z[j]

            # positive: y=1
            s_pos = float(np.dot(Zi, Zj))
            p_pos = sigmoid(s_pos)
            g_pos = (1.0 - p_pos)

            grad_i = g_pos * Zj - lambda_reg * Zi
            grad_j = g_pos * Zi - lambda_reg * Zj

            grad_i = np.clip(grad_i, -clip_value, clip_value)
            grad_j = np.clip(grad_j, -clip_value, clip_value)

            Z[i] += lr * grad_i
            Z[j] += lr * grad_j

            # negatives: y=0
            for _ in range(int(neg_per_pos)):
                k = sample_negative(i)
                Zk = Z[k]

                s_neg = float(np.dot(Z[i], Zk))
                p_neg = sigmoid(s_neg)
                g_neg = (0.0 - p_neg)

                grad_i = g_neg * Zk - lambda_reg * Z[i]
                grad_k = g_neg * Z[i] - lambda_reg * Zk

                grad_i = np.clip(grad_i, -clip_value, clip_value)
                grad_k = np.clip(grad_k, -clip_value, clip_value)

                Z[i] += lr * grad_i
                Z[k] += lr * grad_k

        diff = np.linalg.norm(Z - prev_Z)
        if diff < epsilon:
            break
        prev_Z = Z.copy()

    # normalize rows (keeps scores bounded)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    Z = Z / norms

    return Z