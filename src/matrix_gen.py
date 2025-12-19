import numpy as np

def generate_blurring_matrix(n, j, k, symmetric=True):
    """
    Generate the blurring matrix A based on the formula from the project description.
    
    Args:
        n (int): Size of the matrix (n x n).
        j (int): Shift parameter (0 for A_l, 1 for A_r).
        k (int): Window size parameter.
        symmetric (bool): Unused in this specific construction but kept for API compatibility.
        
    Returns:
        numpy.ndarray: The n x n blurring matrix.
    """

    weights = np.arange(k, 0, -1)
    scaling_factor = 2 / (k * (k + 1))
    coeffs = weights * scaling_factor
    
    A = np.zeros((n, n))
    

    for i, c in enumerate(coeffs):
        if i < n:
            diag_matrix = np.diag(np.full(n - i, c), k=i)
            A += diag_matrix
        

    if j == 1:
        last_c = coeffs[-1]
        A += np.diag(np.full(n - 1, last_c), k=-1)
        
    return A
