"""
Tests for the Lanczos algorithm.
"""

import numpy as np

import pytest

from lanczos_trace_estimator import lanczos


def test_dense_lanczos(
    dense_symmetric_matrix,
    random_seed,
):
    """
    Test to make sure that the Lanczos algorithm is working as expected for dense matrices.
    """
    M = dense_symmetric_matrix
    n = M.shape[0]
    rng = np.random.default_rng(random_seed)
    b = rng.standard_normal(n)
    V, alpha, beta = lanczos(M, b, reorth=True)
    T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    np.testing.assert_allclose(T, V.T @ (M @ V), atol=1e-3, rtol=1e-3)


def test_sparse_lanczos(
    sparse_symmetric_matrix,
    random_seed,
):
    """
    Test to make sure that the Lanczos algorithm is working as expected for dense matrices.
    """
    M = sparse_symmetric_matrix
    n = M.shape[0]
    rng = np.random.default_rng(random_seed)
    b = rng.standard_normal(n)
    V, alpha, beta = lanczos(M, b, reorth=True)
    T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    np.testing.assert_allclose(T, V.T @ (M @ V), atol=1e-3, rtol=1e-3)


def test_linear_operator_lanczos(
    symmetric_linear_operator,
    random_seed,
):
    """
    Test to make sure that the Lanczos algorithm is working as expected for linear operators.
    """
    M = symmetric_linear_operator
    n = M.shape[0]
    rng = np.random.default_rng(random_seed)
    b = rng.standard_normal(n)
    V, alpha, beta = lanczos(M, b, reorth=True)
    T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    np.testing.assert_allclose(T, V.T @ (M @ V), atol=1e-3, rtol=1e-3)


def test_low_rank_lanczos(low_rank_matrix, random_seed):
    """
    Test to make sure the Lanczos algorithm stops appropriately when the matrix
    is of lower rank than the number of iterations.
    """
    M = low_rank_matrix
    rank = np.linalg.matrix_rank(M)
    rng = np.random.default_rng(random_seed)
    b = rng.standard_normal(M.shape[0])
    V, alpha, beta = lanczos(M, b, k=2 * rank, reorth=True)
    T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
    np.testing.assert_allclose(V.T @ (M @ V), T, atol=1e-3, rtol=1e-3)
    assert abs(len(beta) - rank) <= 1
