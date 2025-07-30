from typing import Protocol


class Recommender(Protocol):
    """
    A protocol that defines the interface for recommendation systems.
    """

    def recommend(self, user_id: int) -> dict[int, float]:
        """
        Recommend items.

        Args:
            user_id (int): The ID of the user for whom to recommend items.

        Returns:
            dict: A dictionary mapping recommended item IDs to their corresponding scores.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
