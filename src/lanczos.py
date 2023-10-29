import numpy as np
from scipy.linalg import eigh_tridiagonal

from notation import linear_operator


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


def fA_b(f, A, b, k, reorth=False):
    V, alpha, beta = lanczos(A, b, k, reorth=reorth)
    T = sps.diags([beta[1:k], alpha, beta[1:k]], [-1, 0, 1])
    T = np.diag(alpha) + np.diag(beta[1:k], 1) + np.diag(beta[1:k], -1)
    L, U = np.linalg.eigh(T)
    ebeta = np.zeros(T.shape[0])
    ebeta[0] = beta[0]
    return np.linalg.multi_dot([V[:, :k], U, np.diag(f(L)), U.T, ebeta])
