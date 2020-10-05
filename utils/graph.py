from collections import defaultdict
from .tensor import normalize, csr_to_torch
from fb15k import get_classes
from scipy import sparse

def build_adjacency_tensor(triples):
    """
    Build an adjacency tensor from a list of triples

    `triples` is a list of string tuples `(head, relation, tail)`. Let R be the number of distinct relations,
    and E the number of entities (heads and tails). Then, the adjacency tensor is a list [T1, T2, ..., TR], with
    each Ti a E x E normalized tensor:
    Ti = Di^-1 * Ai
    With Aijk = 1 if (e_k, r_i, e_j) is in `triples`, and 0 otherwise
    And Di a diagonal matrix containing the in-degree of each entity

    The function also return the class of each entity, and mappings from entities to theirs ids and from
    relations to their ids.
    """
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