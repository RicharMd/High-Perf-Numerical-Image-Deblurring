import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def load_image(filepath, grayscale=True):
    """
    Load an image from file.
    
    Args:
        filepath (str): Path to the image file.
        grayscale (bool): Whether to convert to grayscale.
        
    Returns:
        numpy.ndarray: Image data as a float array in [0, 1].
    """
    try:
        img = Image.open(filepath)
        if grayscale:
            img = img.convert('L')
        img_array = np.array(img).astype(np.float64) / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading image {filepath}: {e}")
        return None

def save_image(image_array, filepath):
    """
    Save a numpy array as an image.
    
    Args:
        image_array (numpy.ndarray): Image data (values should be roughly in [0, 1]).
        filepath (str): Output path.
    """
    img_clipped = np.clip(image_array, 0, 1)
    img_uint8 = (img_clipped * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8)
    img.save(filepath)
    print(f"Image saved to {filepath}")

def compute_psnr(img1, img2):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1, img2 (numpy.ndarray): Images to compare. Should have same shape.
        
    Returns:
        float: PSNR value in dB.
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
        
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    return 10 * np.log10(1.0 / mse)

def compute_relative_error(X_true, X_rec):
    """
    Compute relative forward error using Frobenius norm.
    
    Args:
        X_true (numpy.ndarray): True/original image.
        X_rec (numpy.ndarray): Reconstructed image.
        
    Returns:
        float: Relative error = ||X_rec - X_true||_F / ||X_true||_F
    """
    if X_true.shape != X_rec.shape:
        raise ValueError("Images must have the same dimensions")
    
    numerator = np.linalg.norm(X_rec - X_true, ord='fro')
    denominator = np.linalg.norm(X_true, ord='fro')
    
    if denominator == 0:
        return float('inf') if numerator > 0 else 0.0
    
    return numerator / denominator

def display_images(images, titles=None, figsize=(15, 5)):
    """
    Display a list of images side by side.
    
    Args:
        images (list): List of numpy arrays.
        titles (list): List of title strings.
    """
    n = len(images)
    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap='gray')
        if titles and i < len(titles):
            plt.title(titles[i])
        plt.axis('off')
    plt.show()

