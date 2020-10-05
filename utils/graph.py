from collections import defaultdict
from .tensor import normalize, csr_to_torch
from fb15k import get_classes
from scipy import sparse

def build_adjacency_tensor(triples):
    hs, rs, ts = zip(*triples)
    entities = {e: i for i, e in enumerate(set(hs) | set(ts))}
    rev_entities = {i: e for e, i in entities.items()}
    relations = {r: i for i, r in enumerate(set(rs))}
    nr, ne = len(relations), len(entities)
    sorted_triples = defaultdict(list)
    for h, r, t in triples:
        sorted_triples[relations[r]].append((entities[t], entities[h]))
    A = []
    for r, coords in sorted_triples.items():
        row_inds, col_inds = zip(*coords)
        data = [1] * len(coords)
        a = sparse.csr_matrix((data, (row_inds, col_inds)), shape=(ne, ne))
        a = normalize(a)
        a = csr_to_torch(a)
        A.append(a)
    e2c = get_classes(triples)
    classes = [e2c.get(rev_entities[i], "unknown") for i in range(len(rev_entities))]
    return A, classes, entities, relations