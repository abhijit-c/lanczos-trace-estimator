"""
Module to approximate tr[f(A)] using Lanczos + Hutchinson.
"""
import numpy as np
import scipy as sp
from scipy.linalg import eigh_tridiagonal, eigh

from lanczos import lanczos

from notation import linear_operator
from typing import Callable


def fA_b(
    f: Callable[[np.ndarray], np.ndarray],
    A: linear_operator,
    b: np.ndarray,
    **lanczos_opts,
) -> np.ndarray:
    """Appoximate f(A)b for symmetric A using the Lanczos algorithm, see [1].

    [1]: Musco, C., Musco, C., & Sidford, A. (2018). Stability of the Lanczos Method for
        Matrix Function Approximation. In Proceedings of the Twenty-Ninth Annual
        ACM-SIAM Symposium on Discrete Algorithms (pp. 1605–1624). Society for
        Industrial and Applied Mathematics. https://doi.org/10.1137/1.9781611975031.105

    Args:
        f (Callable[[np.ndarray], np.ndarray]): Function to apply to A.
        A (linear_operator): Symmetric matrix to apply f to.
        b (np.ndarray): Vector for which to compute f(A)b.

    Keyword Args:
        reorth (bool): Whether to reorthogonalize the Lanczos vectors. This can be
            quite expensive, but may be necessary for stability. Defaults to False.
        k (int): Number of Lanczos iterations. Defaults to 25.

    Returns:
        np.ndarray: Approximation to f(A)b.
    """
    V, alpha, beta = lanczos(A, b, **lanczos_opts)
    k = len(alpha)
    L, U = eigh_tridiagonal(alpha, beta)
    ebeta = np.zeros(k)
    ebeta[0] = np.linalg.norm(b)
    return V @ (U @ (f(L) * (U.T @ (ebeta))))


# def hutchinson(A, f, n_hutch, k_lanczos, rng, reorth=False, verbose=False):
#    n = A.shape[0]
#    X = rng.standard_normal((n, n_hutch))
#    samps = np.zeros(n_hutch)
#    for i in range(n_hutch):
#        fA_Xi = fA_b(f, A, X[:, i], k_lanczos, reorth=reorth)
#        samps[i] = np.inner(X[:, i], fA_Xi)
#    return samps.mean(), samps
