import pandas as pd
import os
from scipy import io
import time
import numpy as np
import matplotlib.image as mpimg
import math
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular
from numba import jit

def output_csv(original_img_list, recover_img_list, running_time):
    score = 0
    for i in range(len(original_img_list)):
        original_img = original_img_list[i]
        recover_img = recover_img_list[i]
        score += np.abs(original_img-recover_img).mean()
        print('error_0' + str(i+1) + ':' + str(score))
    if score >= 30:
        score = 1
        print('inaccurate deblurring')
    else:
        score = 0
    final_score = (0.25*score+1)*running_time
    print('overall_score:'+ str(final_score))
    df_rec = pd.DataFrame({'Id': ['result'], 'Predicted': [final_score]})
    df_rec.to_csv('./submission_givens.csv', index=False)

def show_image(img,type):
    pass

@jit(nopython=True, cache=True, fastmath=True)
def givens_qr_solve(A, B):
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

def your_soln_func(B, A_l, A_r):
    # Numba 
    B_contig = np.ascontiguousarray(B)
    A_l_contig = np.ascontiguousarray(A_l)
    A_r_T_contig = np.ascontiguousarray(A_r.T)
    
    R_l, QtB = givens_qr_solve(A_l_contig, B_contig)
    Y = solve_triangular(R_l, QtB, lower=False)
    
    Y_T_contig = np.ascontiguousarray(Y.T)
    R_r, QtYT = givens_qr_solve(A_r_T_contig, Y_T_contig)
    XT = solve_triangular(R_r, QtYT, lower=False)
    
    return XT.T

def revcover_img(dataset_dir, your_soln_func):
    revcover_img_list = []
    original_img_list = []
    
    # Warmup
    print("Warming up JIT...")
    dummy_N = 50
    dummy_A = np.eye(dummy_N) + np.diag(np.ones(dummy_N-1), 1)
    dummy_B = np.random.rand(dummy_N, dummy_N)
    givens_qr_solve(np.ascontiguousarray(dummy_A), np.ascontiguousarray(dummy_B))
    print("JIT Warmup done.")
    
    s = time.time()
    p_list = ['01', '02', '03']
    
    for p in p_list:
        try:
            A_l = io.loadmat(os.path.join(dataset_dir,'kaggle_'+ p + '_Ab' '.mat'))['Ab']
            A_r = A_l
            B = io.loadmat(os.path.join(dataset_dir,'kaggle_'+ p + '_blurry' '.mat'))['B']
            original_img = io.loadmat(os.path.join(dataset_dir, 'kaggle_'+ p + '_original' '.mat'))['img']
            
            # Skip plotting
            
            # show_image(original_img,p+'_original')
            original_img_list.append(original_img)
            # show_image(B,p + '_blurry')
            
            recovery_img = your_soln_func(B,A_l,A_r)
            # show_image(recovery_img,p + '_fixed')
            revcover_img_list.append(recovery_img)
        except FileNotFoundError:
            print(f"Skipping {p} as files are missing")
            continue

    running_time = time.time() - s
    output_csv(original_img_list, revcover_img_list, running_time)

if __name__ == '__main__':
    dataset_dir = '/kaggle/input/25-fall-dda-3005-course-project' 
    if not os.path.exists(dataset_dir):
        # Just a placeholder warning
        pass
    else:
        revcover_img(dataset_dir, your_soln_func)

