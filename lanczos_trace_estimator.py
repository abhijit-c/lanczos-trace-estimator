"""
Provides functionality for
- Lanczos algorithm for tridiagonalization of a symmetric matrix.
- Approximating f(A)b for symmetric A using the Lanczos algorithm
- Approximating tr[f(A)] using Hutchinson + Lanczos
"""

import numpy as np
import scipy as sp
from scipy.linalg import eigh_tridiagonal, eigh

from typing import Callable

linear_operator = np.ndarray | sp.sparse.spmatrix | sp.sparse.linalg.LinearOperator


def lanczos(
    A: linear_operator,
    b: np.ndarray,
    k: int = 25,
    reorth: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lanczos algorithm for tridiagonalization of a symmetric matrix.

    Using the Lanczos algorithm, compute the tridiagonal matrix T such that T = V^T A V,
    where V is an orthogonal matrix. if A is in R^{n x n}, then T is in R^{k x k}
    and V is in R^{n x k}. Note, V is computed in its entirety, so if n is extremely
    large this may be infeasible. The eigenvalues of T interlace those of A and can be
    viewed as an approximation to the spectrum of A.

    Args:
        A (linear_operator): Symmetric matrix to tri-diagonalize.
        b (ndarray): Initial vector.
        k (int): Number of Lanczos iterations. Defaults to 25.
        reorth (bool): Whether to reorthogonalize the Lanczos vectors. This can be
            quite expensive, but may be necessary for stability. Defaults to False.

    Returns:
        tuple[ndarray, ndarray, ndarray]: Tuple of the orthogonal matrix V, the
            diagonal of T, and the subdiagonal of T.
    """
    n = A.shape[0]
    V = np.zeros((n, k + 1))
    alpha = np.zeros(k)
    beta = np.zeros(k)

    V[:, 0] = b / np.linalg.norm(b)
    beta[0] = 0
    V[:, 1] = A @ V[:, 0]
    alpha[0] = np.inner(V[:, 1], V[:, 0])
    for j in range(k - 1):
        V[:, j + 1] = V[:, j + 1] - alpha[j] * V[:, j]
        if reorth:  # Full reorthogonalization, expensive
            Vj = V[:, :j]
            V[:, j + 1] -= np.dot(Vj, np.dot(Vj.T, V[:, j + 1]))
        beta[j + 1] = np.linalg.norm(V[:, j + 1])
        if np.isclose(beta[j + 1], 0.0):
            return V[:, : j + 1], alpha[: j + 1], beta[1 : j + 1]
        V[:, j + 1] /= beta[j + 1]
        V[:, j + 2] = A @ V[:, j + 1] - beta[j + 1] * V[:, j]
        alpha[j + 1] = np.inner(V[:, j + 2], V[:, j + 1])
    return V[:, :-1], alpha, beta[1:]


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


def hutchinson(
    A: linear_operator,
    f: Callable[[np.ndarray], np.ndarray],
    n: int = 30,
    seed=None,
    **lanczos_opts,
) -> tuple[float, np.ndarray]:
    """Approximate tr[f(A)] using Hutchinson + Lanczos.

    Using the Hutchinson algorithm to approximate the trace and the Lanczos algrorithm
    to approximate f(A)b, we can approximate tr[f(A)]. See the following paper for
    details:

    Ubaru, S., Chen, J., & Saad, Y. (2017). Fast Estimation of $tr(f(A))$ via Stochastic
    Lanczos Quadrature. In SIAM Journal on Matrix Analysis and Applications (Vol. 38,
    Issue 4, pp. 1075–1099). Society for Industrial & Applied Mathematics (SIAM).
    https://doi.org/10.1137/16m1104974

    Args:
        A (linear_operator): Symmetric matrix to apply f to.
        f (Callable[[np.ndarray], np.ndarray]): Function to apply to A.

    Keyword Args:
        n (int): Number of samples to use in the Hutchinson algorithm. Defaults to 30.
        seed: Random seed for the Hutchinson algorithm. Will be passed to
            np.random.default_rng, so can be an int, array_like, or BitGenerator.
            Defaults to None.
        reorth (bool): Whether to reorthogonalize the Lanczos vectors. This can be
            quite expensive, but may be necessary for stability. Defaults to False.
        k (int): Number of Lanczos iterations. Defaults to 25.

    Returns:
        tuple[float, np.ndarray]: Tuple of the estimate of tr[f(A)] and the samples
            used to compute the estimate.
    """
    rng = np.random.default_rng(seed)
    # X = rng.standard_normal((A.shape[0], n))
    X = 2 * (rng.binomial(1, 0.5, size=(A.shape[0], n)) - 0.5)
    samps = np.zeros(n)
    for i in range(n):
        samps[i] = np.inner(X[:, i], fA_b(f, A, X[:, i], **lanczos_opts))
    return samps.mean(), samps
