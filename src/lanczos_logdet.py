import numpy as np
import scipy as sp
import scipy.sparse as sps


def lanczos(A, b, k, reorth=False):
    n = A.shape[0]
    V = np.zeros((n, k + 1), dtype="d")
    alpha = np.zeros((k,), dtype="d")
    beta = np.zeros((k + 1,), dtype="d")

    beta[0] = np.linalg.norm(b)
    vjm1 = 0.0
    v = b / beta[0]
    V[:, 0] = np.copy(v)
    for j in range(k):
        w = A @ v - beta[j] * vjm1
        vjm1 = np.copy(v)
        # Orthogonalize
        alpha[j] = np.inner(w, v)
        w = w - alpha[j] * v
        if reorth:
            Vj = V[:, :j]
            w -= np.dot(Vj, np.dot(Vj.T, w))
        beta[j + 1] = np.linalg.norm(w)
        v = w / beta[j + 1]
        # Store current iterate
        V[:, j + 1] = np.copy(v)
    return V, alpha, beta


def fA_b(f, A, b, k, reorth=False):
    V, alpha, beta = lanczos(A, b, k, reorth=reorth)
    T = sps.diags([beta[1:k], alpha, beta[1:k]], [-1, 0, 1])
    T = np.diag(alpha) + np.diag(beta[1:k], 1) + np.diag(beta[1:k], -1)
    L, U = np.linalg.eigh(T)
    ebeta = np.zeros(T.shape[0])
    ebeta[0] = beta[0]
    return np.linalg.multi_dot([V[:, :k], U, np.diag(f(L)), U.T, ebeta])


def hutchinson(A, f, n_hutch, k_lanczos, rng, reorth=False, verbose=False):
    n = A.shape[0]
    X = rng.standard_normal((n, n_hutch))
    samps = np.zeros(n_hutch)
    for i in range(n_hutch):
        fA_Xi = fA_b(f, A, X[:, i], k_lanczos, reorth=reorth)
        samps[i] = np.inner(X[:, i], fA_Xi)
    return samps.mean(), samps
