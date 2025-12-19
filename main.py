import os
import time
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_image, save_image, compute_psnr, compute_relative_error, display_images
from src.matrix_gen import generate_blurring_matrix
from src.solvers import solve_image_system

def main():
    # Setup paths
    image_paths = [
        "selected_image/1500_m6_original.png",
        "selected_image/512_car_original.png"
    ]
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Define two different kernel parameter sets
    kernel_configs = [
        {'name': 'config1', 'A_l': {'j': 0, 'k': 12}, 'A_r': {'j': 1, 'k': 36}},
        {'name': 'config2', 'A_l': {'j': 0, 'k': 8}, 'A_r': {'j': 1, 'k': 24}}
    ]
    
    methods = [
        ('LU', 'lu'),
        ('Standard QR', 'qr'),
        ('My Householder QR', 'my_qr'),
        ('Numba Givens QR', 'numba_qr'),
        ('FFT Deconv', 'fft')
    ]
        
    for img_path in image_paths:
        print(f"\n{'='*60}")
        print(f"Processing {img_path}...")
        print(f"{'='*60}")
        
        # 1. Load Image
        X = load_image(img_path)
        if X is None:
            continue
        
        n = X.shape[0]
        print(f"Image size: {n}x{n}")
        img_name = os.path.basename(img_path).split('.')[0]
        
        # Store results for both configurations
        all_results = []
        
        for config_idx, config in enumerate(kernel_configs):
            print(f"\n--- Kernel Configuration {config_idx + 1}: {config['name']} ---")
            print(f"  A_l: j={config['A_l']['j']}, k={config['A_l']['k']}")
            print(f"  A_r: j={config['A_r']['j']}, k={config['A_r']['k']}")
            
            # 2. Generate Blurring Kernels
            A_l = generate_blurring_matrix(n, j=config['A_l']['j'], k=config['A_l']['k'])
            A_r = generate_blurring_matrix(n, j=config['A_r']['j'], k=config['A_r']['k'])
            
            # 3. Generate Blurry Image
            B = A_l @ X @ A_r
            save_image(B, os.path.join(output_dir, f"{img_name}_blurred_{config['name']}.png"))
        
        # 4. Run all solvers
            config_results = {'config': config['name'], 'blurred': B, 'reconstructions': {}}
            
        for name, key in methods:
            try:
                    print(f"  Reconstructing with {name}...")
                
                # Warmup for numba
                if key == 'numba_qr':
                    dummy_A = np.eye(10) + np.diag(np.ones(9), 1)
                    dummy_B = np.random.rand(10, 10)
                    try:
                        solve_image_system(dummy_A, dummy_A, dummy_B, method='numba_qr')
                    except:
                        pass

                start_time = time.time()
                X_rec = solve_image_system(A_l, A_r, B, method=key)
                dt = time.time() - start_time
                psnr = compute_psnr(X, X_rec)
                    rel_error = compute_relative_error(X, X_rec)
                    print(f"    {name}: Time={dt:.4f}s, PSNR={psnr:.2f}dB, Rel Error={rel_error:.2e}")
                
                    save_image(X_rec, os.path.join(output_dir, f"{img_name}_rec_{config['name']}_{key}.png"))
                    config_results['reconstructions'][name] = {
                        'image': X_rec,
                        'time': dt,
                        'psnr': psnr,
                        'rel_error': rel_error
                    }
                
            except Exception as e:
                    print(f"    Method {name} failed: {e}")
                    config_results['reconstructions'][name] = {
                        'image': B,
                        'time': 0,
                        'psnr': 0,
                        'rel_error': float('inf')
                    }
            
            all_results.append(config_results)
        
        # 5. Display comparison (side by side for each method, stacked for two configs)
        print(f"\nGenerating comparison visualization...")
        n_rows = len(methods) + 2  # Original + Blurred + Methods
        n_cols = 2
        fig = plt.figure(figsize=(12, 4 * n_rows))
        
        # Row 0: Original image (span both columns)
        ax_orig = plt.subplot2grid((n_rows, n_cols), (0, 0), colspan=n_cols, fig=fig)
        ax_orig.imshow(X, cmap='gray')
        ax_orig.set_title(f'Original Image: {img_name}', fontsize=14, fontweight='bold')
        ax_orig.axis('off')
        
        # Rows 1 to n_rows-1: Blurred and reconstructions
        row_idx = 1
        for method_name, method_key in [('Blurred', None)] + methods:
            for col_idx, config_result in enumerate(all_results):
                ax = plt.subplot2grid((n_rows, n_cols), (row_idx, col_idx), fig=fig)
                
                if method_key is None:  # Blurred image
                    img = config_result['blurred']
                    title = f"Blurred ({config_result['config']})"
                else:  # Reconstructed image
                    if method_name in config_result['reconstructions']:
                        rec_data = config_result['reconstructions'][method_name]
                        img = rec_data['image']
                        title = f"{method_name} ({config_result['config']})\n"
                        title += f"Time: {rec_data['time']:.3f}s, PSNR: {rec_data['psnr']:.2f}dB\n"
                        title += f"Rel Error: {rec_data['rel_error']:.2e}"
                    else:
                        img = config_result['blurred']
                        title = f"{method_name} ({config_result['config']}) - Failed"
                
                ax.imshow(img, cmap='gray')
                ax.set_title(title, fontsize=10)
                ax.axis('off')
            
            row_idx += 1
        
        plt.tight_layout()
        comparison_file = os.path.join(output_dir, f"{img_name}_comparison.png")
        plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {comparison_file}")
        plt.close()

if __name__ == "__main__":
    main()
