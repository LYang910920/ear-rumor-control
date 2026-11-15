"""Complete contact networks and the four-community synthetic example."""
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
from scipy import sparse
from ear.aggregation import capped_contacts, quotient_from_sparse


def synthetic_network(n=10000, seed=20260907):
    rng = np.random.default_rng(seed)
    blocks = np.arange(n) * 4 // n
    members = [np.flatnonzero(blocks == b) for b in range(4)]
    rows, cols, br, bc = [], [], [], []
    for i in range(n):
        same = members[blocks[i]]
        target = np.r_[rng.choice(same, 8, replace=True),
                       rng.integers(n, size=4), (i + 1) % n]
        rows.extend([i] * len(target)); cols.extend(target.tolist())
        target = np.r_[rng.choice(same, 4, replace=True),
                       rng.integers(n, size=6), (i + 1) % n]
        br.extend([i] * len(target)); bc.extend(target.tolist())
    A = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    B = sparse.coo_matrix((np.ones(len(br)), (br, bc)), shape=(n, n)).tocsr()
    for matrix in (A, B):
        matrix.setdiag(0); matrix.eliminate_zeros(); matrix.data[:] = 1.
    return A, B


def read_pairs(folder, layer):
    name = f"higgs-{layer}_network.edgelist"
    path = Path(folder) / name
    if not path.exists():
        path = path.with_suffix(path.suffix + ".gz")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve("https://snap.stanford.edu/data/" + path.name, path)
    return np.loadtxt(path, dtype=np.int64, usecols=(0, 1), ndmin=2)


def binary_matrix(pairs, ids, reverse=False):
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    index = np.searchsorted(ids, pairs)
    row, col = (index[:, 1], index[:, 0]) if reverse else (index[:, 0], index[:, 1])
    matrix = sparse.coo_matrix((np.ones(len(pairs)), (row, col)),
                              shape=(len(ids), len(ids))).tocsr()
    matrix.sum_duplicates(); matrix.data[:] = 1.; matrix.sort_indices()
    return matrix


def prepare(network, output, raw="data/raw"):
    """Build normalized node contacts; matrix [i,j] carries influence j to i."""
    if network == "synthetic":
        A, B = synthetic_network()
    else:
        layer = "retweet" if network == "rt_mt" else "social"
        first, mentions = read_pairs(raw, layer), read_pairs(raw, "mention")
        ids = np.union1d(first.ravel(), mentions.ravel())
        A = binary_matrix(first, ids)
        B = binary_matrix(mentions, ids, reverse=True)
    normalized = []
    for matrix in (A, B):
        degree = np.asarray(matrix.getnnz(axis=1))
        reference = float(max(1, np.median(degree[degree > 0])))
        normalized.append(capped_contacts(matrix, reference, 5.))
    A, B = normalized
    q = quotient_from_sparse(A, B, max_groups=32, compute_residual=False)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output / "A.npz", A)
    sparse.save_npz(output / "B.npz", B)
    np.savez_compressed(output / "groups.npz", A=q.A, B=q.B,
                        pi=q.pi, membership=q.membership)
    print(f"{network}: {A.shape[0]} users, {A.nnz}/{B.nnz} contacts, {len(q.pi)} groups")


def contacts(folder):
    folder = Path(folder)
    return sparse.load_npz(folder / "A.npz"), sparse.load_npz(folder / "B.npz")
