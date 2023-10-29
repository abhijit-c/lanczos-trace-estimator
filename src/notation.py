"""
This file contains type aliases for the project.
"""
import numpy as np
import scipy as sp

linear_operator = np.ndarray | sp.sparse.spmatrix | sp.sparse.linalg.LinearOperator
