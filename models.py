from typing import Callable, Literal
from collections import defaultdict
import hnswlib
import numpy as np


class BaseRecommender:
    def recommend(self, history_ids: list[int]) -> dict[int, float]:
        """
        Recommend items.

        Args:
            history_ids (list[int]): List of item IDs representing the user's history.

        Returns:
            tuple: A tuple containing a list of recommended item IDs and their corresponding scores.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class Index:
    def __init__(
        self,
        embeddings: np.ndarray,
        space: Literal["l2", "ip", "cosine"],
        ef=50,
    ):
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
            sim = 1 - d
            idx = int(i)
            scores[idx] = max(scores[idx], sim)

        return scores


class EmbeddingRecommender(BaseRecommender):
    def __init__(
        self,
        index: Index,
        get_history_embeddings: Callable[[list[int]], np.ndarray],
        k: int,
        sample_weight: float,
    ):
        self.index = index
        self.k = k
        self.sample_weight = sample_weight
        self.get_history_embeddings = get_history_embeddings

    def recommend(self, history_ids: list[int]) -> dict[int, float]:
        history_embeddings = self.get_history_embeddings(history_ids)
        scores = self.index.query(history_embeddings, int(self.sample_weight * self.k))
        print(f"Found {len(scores)} recommendations for {len(history_embeddings)} history items")
        return scores
