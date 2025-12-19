import numpy as np
from scipy.linalg import lu, solve_triangular, qr
from scipy.fft import fft2, ifft2, fft

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: Numba not found. Accelerated solvers will not be available.")

def householder_qr(A):
    """
    Perform Householder QR factorization.
    Returns R and Householder vectors W. Q is not explicitly formed for efficiency.
    """
    m, n = A.shape
    R = A.copy()
    W = np.zeros((m, n))
    
    for k in range(n):
        v = R[k:m, k].copy()
        
        e1 = np.zeros_like(v)
        e1[0] = 1.0
        
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            continue 
            
        if v[0] > 0:
            v = v + v_norm * e1
        else:
            v = v - v_norm * e1
            
        v = v / np.linalg.norm(v)
        
        W[k:m, k] = v
        
        v_reshaped = v.reshape(-1, 1)
        R_sub = R[k:m, k:n]
        R[k:m, k:n] = R_sub - 2 * np.dot(v_reshaped, np.dot(v_reshaped.T, R_sub))
        
    return R, W

def apply_qt_from_householder(W, B):
    """
    Efficiently compute Q^T * B without explicitly forming Q.
    """
    m, n = W.shape
    result = B.copy()
    
    for k in range(n):
        v = W[k:m, k]
        if np.all(v == 0):
            continue
        v_reshaped = v.reshape(-1, 1)
        result_sub = result[k:m, :]
        result[k:m, :] = result_sub - 2 * np.dot(v_reshaped, np.dot(v_reshaped.T, result_sub))
    
    return result

if HAS_NUMBA:
    @jit(nopython=True, cache=True, fastmath=True)
    def givens_qr_solve_numba(A, B):
        m, n = A.shape
        R = A.copy()
        QtB = B.copy()
        cols_B = B.shape[1]
        
        BANDWIDTH = 50
        
        for j in range(n):
            end_row = min(j + BANDWIDTH, m)
            for i in range(j+1, end_row):
                if R[i, j] != 0:
                    pivot = R[j, j]
                    target = R[i, j]
                    
                    if target == 0:
                        c, s = 1.0, 0.0
                    elif abs(target) > abs(pivot):
                        tau = -pivot / target
                        s = 1.0 / np.sqrt(1 + tau*tau)
                        c = s * tau
                    else:
                        tau = -target / pivot
                        c = 1.0 / np.sqrt(1 + tau*tau)
                        s = c * tau
                    
                    # Update R
                    update_end = min(j + BANDWIDTH + 10, n)
                    for k in range(j, update_end):
                        val_j = R[j, k]
                        val_i = R[i, k]
                        R[j, k] = c * val_j - s * val_i
                        R[i, k] = s * val_j + c * val_i
                        
                    # Update B
                    for k in range(cols_B):
                        val_j = QtB[j, k]
                        val_i = QtB[i, k]
                        QtB[j, k] = c * val_j - s * val_i
                        QtB[i, k] = s * val_j + c * val_i
                        
        return R, QtB

def solve_fft(A_l, A_r, B):
    rows, cols = B.shape
    mid_idx = rows // 2
    
    col_kernel = A_l[:, mid_idx].flatten()
    col_kernel_shifted = np.roll(col_kernel, -mid_idx)
    
    row_kernel = A_r[mid_idx, :].flatten()
    row_kernel_shifted = np.roll(row_kernel, -mid_idx)

    B_fft = fft2(B)
    H_l = fft(col_kernel_shifted, n=rows).reshape(-1, 1)
    H_r = fft(row_kernel_shifted, n=cols).reshape(1, -1)

    epsilon = 1e-3 
    X_fft = B_fft / (H_l * H_r + epsilon)
    
    X_rec = np.real(ifft2(X_fft))
    return np.clip(X_rec, 0, 1)

def solve_image_system(A_l, A_r, B, method='lu'):
    """
    Solve the system A_l * X * A_r = B for X.
    Supported methods: 'lu', 'qr', 'my_qr', 'numba_qr', 'fft'
    """
    if method == 'lu':
        P1, L1, U1 = lu(A_l)
        P2, L2, U2 = lu(A_r.T)
        Y1 = solve_triangular(L1, P1.T @ B, lower=True)
        X1 = solve_triangular(U1, Y1)
        Y2 = solve_triangular(L2, P2.T @ X1.T, lower=True)
        X2 = solve_triangular(U2, Y2)
        X = X2.T
    
    elif method == 'qr':
        Q1, R1 = qr(A_l)
        Q2, R2 = qr(A_r.T)
        Y1 = solve_triangular(R1, Q1.T @ B, lower=False)
        Y2 = solve_triangular(R2, Q2.T @ Y1.T, lower=False)
        X = Y2.T
        
    elif method == 'my_qr':
        R_l, W_l = householder_qr(A_l)
        R_r, W_r = householder_qr(A_r.T)
        
        QtB = apply_qt_from_householder(W_l, B)
        Y1 = solve_triangular(R_l, QtB, lower=False)
        
        QtY1T = apply_qt_from_householder(W_r, Y1.T)
        Y2 = solve_triangular(R_r, QtY1T, lower=False)
        X = Y2.T
        
    elif method == 'numba_qr':
        if not HAS_NUMBA:
            raise RuntimeError("Numba is not installed")
        
        B_contig = np.ascontiguousarray(B)
        A_l_contig = np.ascontiguousarray(A_l)
        A_r_T_contig = np.ascontiguousarray(A_r.T)
        
        R_l, QtB = givens_qr_solve_numba(A_l_contig, B_contig)
        Y = solve_triangular(R_l, QtB, lower=False)
        
        Y_T_contig = np.ascontiguousarray(Y.T)
        R_r, QtYT = givens_qr_solve_numba(A_r_T_contig, Y_T_contig)
        XT = solve_triangular(R_r, QtYT, lower=False)
        X = XT.T
        
    elif method == 'fft':
        X = solve_fft(A_l, A_r, B)
        
    else:
        raise ValueError(f"Unknown method: {method}")
        
    return X
