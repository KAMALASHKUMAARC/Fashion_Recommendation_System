import time
import numpy as np
from contextlib import contextmanager


@contextmanager
def timer(name: str):
    t0 = time.time()
    yield
    print(f"[{name}] {time.time() - t0:.2f}s")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def recall_at_k(retrieved_ids, relevant_ids, k: int) -> float:
    top = set(retrieved_ids[:k])
    return 1.0 if top & set(relevant_ids) else 0.0


def average_precision_at_k(retrieved_ids, relevant_ids, k: int) -> float:
    relevant = set(relevant_ids)
    hits, precisions = 0, []
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant:
            hits += 1
            precisions.append(hits / (i + 1))
    if not precisions:
        return 0.0
    return float(np.mean(precisions))


def map_at_k(all_retrieved, all_relevant, k: int) -> float:
    aps = [average_precision_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)]
    return float(np.mean(aps))