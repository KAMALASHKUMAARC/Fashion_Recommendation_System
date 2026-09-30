import faiss
import numpy as np
import pickle

from config import NLIST, NPROBE, EMBEDDING_DIM


class FAISSIndex:
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.index = None
        self.id_map = None

    def build(self, embeddings: np.ndarray, id_map: np.ndarray):
        assert embeddings.shape[1] == self.dim
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        quantizer = faiss.IndexFlatIP(self.dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.dim, NLIST, faiss.METRIC_INNER_PRODUCT)
        self.index.train(embeddings)
        self.index.add(embeddings)
        self.index.nprobe = NPROBE
        self.id_map = np.asarray(id_map)

    def search(self, query_vec: np.ndarray, k: int = 100):
        q = query_vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        ids = self.id_map[I[0]]
        return D[0], ids

    def save(self, path: str):
        faiss.write_index(self.index, str(path))
        with open(str(path) + ".idmap.pkl", "wb") as f:
            pickle.dump(self.id_map, f)

    def load(self, path: str):
        self.index = faiss.read_index(str(path))
        self.index.nprobe = NPROBE
        with open(str(path) + ".idmap.pkl", "rb") as f:
            self.id_map = pickle.load(f)