import numpy as np
import scipy as sp


import pytest

from lanczos import lanczos


def test_lanczos(
    dense_symmetric_matrix,
    sparse_symmetric_matrix,
    symmetric_linear_operator,
):
    """
    Test to make sure that the Lanczos algorithm is working as expected.
    """
    for M in [
        dense_symmetric_matrix,
        sparse_symmetric_matrix,
        symmetric_linear_operator,
    ]:
        n = M.shape[0]
        b = np.random.rand(n)
        V, alpha, beta = lanczos(M, b, reorth=True)
        T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
        np.testing.assert_allclose(T, V.T @ (M @ V), atol=1e-10, rtol=1e-10)


def test_lanczos_early_stop(
    low_rank_matrix,
):
    """
    Test to make sure the Lanczos algorithm stops appropriately when the matrix
    is of lower rank than the number of iterations.
    """
    rank, M = low_rank_matrix
    b = np.random.rand(M.shape[0])
    V, alpha, beta = lanczos(M, b, k=2 * rank, reorth=True)
    T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    np.testing.assert_allclose(T, V.T @ (M @ V), atol=1e-10, rtol=1e-10)
    assert len(beta) == rank
