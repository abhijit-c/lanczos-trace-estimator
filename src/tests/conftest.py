import pytest
import numpy as np
import scipy as sp

RANDOM_SEED = 42


@pytest.fixture
def dense_symmetric_matrix():
    """
    Create a dense symmetric matrix.
    """
    n = 500
    rng = np.random.default_rng(RANDOM_SEED)
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2


@pytest.fixture
def sparse_symmetric_matrix():
    """
    Create a sparse symmetric matrix.
    """
    n = 1000
    rng = np.random.default_rng(RANDOM_SEED)
    d0 = rng.standard_normal(n)
    d1 = rng.standard_normal(n - 1)
    return sp.sparse.diags([d1, d0, d1], [-1, 0, 1])


@pytest.fixture
def symmetric_linear_operator(dense_symmetric_matrix):
    """
    Create a linear operator from a dense symmetric matrix.
    """
    return sp.sparse.linalg.LinearOperator(
        dense_symmetric_matrix.shape, matvec=lambda x: dense_symmetric_matrix @ x
    )


@pytest.fixture
def low_rank_matrix():
    """
    Create a low rank matrix.
    """
    n = 1000
    rank = 10
    rng = np.random.default_rng(RANDOM_SEED)
    A = rng.standard_normal((n, rank))
    G = rng.standard_normal(rank)
    return rank, A @ np.diag(G) @ A.T
