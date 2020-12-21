#!/usr/bin/env python3
import numpy as np
import faiss
import knn

d = 128
nb = 100000
nq = 10000
np.random.seed(1234)
xb = np.random.random((nb,d)).astype('float32')
xb[:,0] += np.arange(nb) / 1000.
xq = np.random.random((nq,d)).astype('float32')
xq[:,0] += np.arange(nq) / 1000.

index = faiss.IndexFlatL2(d)
print("index.is_trained =", index.is_trained)
index.add(xb)
print("index.ntotal =", index.ntotal)

k = 4
D, I = index.search(xb[:5], k)
print(I)
print(D)
D, I = index.search(xq, k)
print(I[:5])
print(I[-5:])
