import numpy as np

np.set_printoptions(precision=4, suppress=True)


X = np.array([[[0.1, 0.2, 0.3, 0.4],
              [0.5, 0.4, 0.3, 0.2],
              [0.0, 0.1, 0.0, 0.1]]], dtype=np.float32)

Wq = np.array([[0.2, -0.1],
               [0.0, 0.1],
               [0.1, 0.2],
               [-0.1, 0.0]], dtype=np.float32)

Wk = np.array([[0.1, 0.1],
               [0.0,- 0.1],
               [0.2, 0.0],
               [0.0, 0.2]], dtype=np.float32)

Wv = np.array([[0.1, 0.0],
               [-0.1, 0.1],
               [0.2, -0.1],
               [0.0, 0.2]], dtype=np.float32)

# project to q, k, v
Q = X @ Wq
K = X @ Wk
V = X @ Wv

# Q = np.expand_dims(Q, axis=0)
# K = np.expand_dims(K, axis=0)
# V = np.expand_dims(V, axis=0)

print(f"Q shape: {Q.shape} = B, T, d_head")
print(f"K shape: {K.shape} = B, T, d_head")
print(f"V shape: {V.shape} = B, T, d_head")

# scaled dot product attention
scale = 1 / np.sqrt(Q.shape[-1])
scores = (Q @ K.transpose(0, 2, 1)) * scale
# causal mask
causal_mask = np.tril(np.ones((1, 3, 3), dtype=np.bool), k=1)
attn_scores = np.where(causal_mask, -1e9, scores)
# apply softmax
weights: np.ndarray = np.exp(attn_scores - attn_scores.max(axis=-1, keepdims=True))
weights = weights / weights.sum(axis=-1, keepdims=True)
print(f"weights shape: {weights.shape} = B, T, T")
out = weights @ V
print(f"out shape: {out.shape} = B, T, d_head")
