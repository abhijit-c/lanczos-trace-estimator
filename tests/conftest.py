import pytest
import numpy as np
import scipy as sp


@pytest.fixture
def random_seed():
    """
    Create a random seed.
    """
    return 42


@pytest.fixture
def dense_symmetric_matrix(random_seed):
    """
    Create a dense symmetric matrix.
    """
    n = 500
    rng = np.random.default_rng(random_seed)
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2


@pytest.fixture
def sparse_symmetric_matrix(random_seed):
    """
    Create a sparse symmetric matrix.
    """
    n = 1000
    rng = np.random.default_rng(random_seed)
    d0 = rng.standard_normal(n)
    d1 = rng.standard_normal(n - 1)
    return sp.sparse.diags([d1, d0, d1], [-1, 0, 1])


@pytest.fixture
def symmetric_linear_operator(dense_symmetric_matrix, random_seed):
    """
    Create a linear operator from a dense symmetric matrix.
    """
    return sp.sparse.linalg.LinearOperator(
        dense_symmetric_matrix.shape, matvec=lambda x: dense_symmetric_matrix @ x
    )


@pytest.fixture
def low_rank_matrix(random_seed):
    """
    Create a low rank matrix.
    """
    n = 500
    rank = 10
    rng = np.random.default_rng(random_seed)
    A = rng.standard_normal((n, rank))
    return A.T @ A


@pytest.fixture
def positive_definite_matrix(random_seed):
    """
    Create a positive definite matrix.
    """
    n = 500
    rng = np.random.default_rng(random_seed)
    A = rng.standard_normal((n, n))
    A = (A.T + A) / 2
    L, V = np.linalg.eigh(A)
    L = np.linspace(0.5, 1.5, n)
    return V @ np.diag(L) @ V.T
