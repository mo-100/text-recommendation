from typing import Literal
from collections import defaultdict
import hnswlib
import numpy as np


class Index:
    def __init__(self, embeddings: np.ndarray, space: Literal["l2", "ip", "cosine"], ef=50):
        index = hnswlib.Index(space=space, dim=embeddings.shape[1])
        index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
        index.add_items(embeddings, np.arange(len(embeddings)))
        index.set_ef(ef)  # ef should always be > k
        self.index = index

    def query(self, query_embeddings: np.ndarray, k=5) -> dict[int, float]:
        indices, distances = self.index.knn_query(query_embeddings, k)
        # flatten
        indices, distances = indices.flatten(), distances.flatten()

        # get max similarity for each item
        scores: defaultdict[int, float] = defaultdict(float)
        for i, d in zip(indices, distances):
            sim = float(1 - d)
            idx = int(i)
            scores[idx] = max(scores[idx], sim)

        return scores
