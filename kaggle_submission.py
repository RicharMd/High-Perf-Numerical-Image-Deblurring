import pandas as pd
import os
from scipy import io
import time
import numpy as np
import matplotlib.image as mpimg
import math
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fft, ifft

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
    df_rec.to_csv('./submission.csv', index=False)

def show_image(img,type):
    # OPTIMIZATION: Skip plotting to save time!
    pass

def your_soln_func(B, A_l, A_r):
    rows, cols = B.shape
    mid_idx = rows // 2
    

    col_kernel_spatial = A_l[:, mid_idx].flatten()
    col_kernel_shifted = np.roll(col_kernel_spatial, -mid_idx)
    
    row_kernel_spatial = A_r[mid_idx, :].flatten()
    row_kernel_shifted = np.roll(row_kernel_spatial, -mid_idx)

    B_fft = fft2(B)
    
    H_l = fft(col_kernel_shifted, n=rows).reshape(-1, 1)
    H_r = fft(row_kernel_shifted, n=cols).reshape(1, -1)
    

    epsilon = 1e-3 
    X_fft = B_fft / (H_l * H_r + epsilon)
    
    X_rec = np.real(ifft2(X_fft))
    
    return np.clip(X_rec, 0, 1)

def revcover_img(dataset_dir, your_soln_func):
    revcover_img_list = []
    original_img_list = []
    
    s = time.time()
    p_list = ['01', '02', '03']
    
    for p in p_list:
        try:
            # Load data
            A_l = io.loadmat(os.path.join(dataset_dir,'kaggle_'+ p + '_Ab' '.mat'))['Ab']
            A_r = A_l
            B = io.loadmat(os.path.join(dataset_dir,'kaggle_'+ p + '_blurry' '.mat'))['B']
            original_img = io.loadmat(os.path.join(dataset_dir, 'kaggle_'+ p + '_original' '.mat'))['img']
            
            # Skip showing images
            original_img_list.append(original_img)
            
            # Solve
            recovery_img = your_soln_func(B,A_l,A_r)
            revcover_img_list.append(recovery_img)
            
        except FileNotFoundError:
            print(f"Skipping {p} as files are missing")
            continue

    running_time = time.time() - s
    output_csv(original_img_list, revcover_img_list, running_time)

if __name__ == '__main__':
    dataset_dir = '/kaggle/input/25-fall-dda-3005-course-project' 
    if not os.path.exists(dataset_dir):
        print(f"Warning: {dataset_dir} not found. Please set the correct path.")
    else:
        revcover_img(dataset_dir, your_soln_func)
