"""
Tests for the Lanczos algorithm.
"""
import numpy as np
from scipy.linalg import logm

import pytest

from trace_estimation import fA_b


def test_Ab(dense_symmetric_matrix, random_seed):
    """
    Test to see whether the lanczos approximation to Ab is correct.
    """
    M = dense_symmetric_matrix
    rng = np.random.default_rng(random_seed)
    b = rng.standard_normal(M.shape[0])

    np.testing.assert_allclose(fA_b(lambda x: x, M, b), M @ b, rtol=1e-5, atol=1e-5)


def test_AAb(dense_symmetric_matrix, random_seed):
    """
    Test to see whether the lanczos approximation to AAb is correct.
    """
    M = dense_symmetric_matrix
    rng = np.random.default_rng(random_seed)
    b = rng.standard_normal(M.shape[0])

    np.testing.assert_allclose(
        fA_b(lambda x: x * x, M, b), M @ M @ b, rtol=1e-5, atol=1e-5
    )


def test_logAb(positive_definite_matrix, random_seed):
    M = positive_definite_matrix
    rng = np.random.default_rng(random_seed)
    b = rng.standard_normal(M.shape[0])

    np.testing.assert_allclose(fA_b(np.log, M, b), logm(M) @ b, rtol=1e-5, atol=1e-5)
