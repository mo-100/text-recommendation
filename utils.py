import random


def unpack_scores(scores: dict[int, float]) -> tuple[list[int], list[float]]:
    """
    Convert a dictionary of scores to a tuple of lists containing item IDs and their corresponding scores.

    Args:
        scores (dict[int, float]): Dictionary where keys are item IDs and values are their scores.

    Returns:
        tuple: A tuple containing a list of item IDs and a list of their corresponding scores.
    """
    scores_ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    indices, distances = [i for i, _ in scores_ranked], [d for _, d in scores_ranked]
    return indices, distances


def skip_history(scores: dict[int, float], history: list[int]) -> dict[int, float]:
    """
    Remove items from scores that are present in the user's history.

    Args:
        scores (dict[int, float]): Dictionary of scores where keys are item IDs and values are their scores.
        history (list[int]): List of item IDs in the user's history.

    Returns:
        dict[int, float]: Filtered dictionary of scores excluding items in the user's history.
    """
    return {k: v for k, v in scores.items() if k not in history}


def take_top_k(scores: dict[int, float], k: int) -> dict[int, float]:
    """
    Get the top k items from the scores dictionary.

    Args:
        scores (dict[int, float]): Dictionary of scores where keys are item IDs and values are their scores.
        k (int): Number of top items to return.

    Returns:
        dict[int, float]: Dictionary containing the top k items and their scores.
    """
    indices, distances = unpack_scores(scores)
    indices, distances = indices[:k], distances[:k]
    return {index: distance for index, distance in zip(indices, distances)}


def take_random(scores: dict[int, float], k: int) -> dict[int, float]:
    """
    Remove items from scores randomly.

    Args:
        scores (dict[int, float]): Dictionary of scores where keys are item IDs and values are their scores.
        k (int): Number of items to return.
    """
    sampled = random.sample(list(scores.items()), k)
    return {index: distance for index, distance in sampled}
