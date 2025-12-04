"""
DDPM Inference Module
=====================
This module implements the inference logic to generate images
using the DDPM (Denoising Diffusion Probabilistic Model).

Generation process:
1. Starts with random noise
2. Applies iterative denoising process
3. Returns a generated image

Functions:
    - generate_image: Generates an image using DDPM model
    - denoise_step: Applies a denoising step
"""

# Import local modules
from src.config.libraries import *

def generate_image(model, device, image_size=(76, 76), channels=3, num_steps=500):
    """
    Generate an image using the DDPM model through the denoising process.
    
    Args:
        model: Loaded DDPM model
        device: Device (cuda or cpu)
        image_size: Tuple (height, width) for image dimensions
        channels: Number of channels (1 for grayscale, 3 for RGB)
        num_steps: Number of denoising steps
        
    Returns:
        PIL.Image: Generated image
    """
    print(f"Generating image {channels}x{image_size[0]}x{image_size[1]}...")
    # Get n number of Inferences
    xt = model.inference()      # (1, C, H, W) en [0,1]
    img = xt[0].cpu().numpy()         # (C, H, W)
    img = (img * 255).clip(0, 255).astype(np.uint8)  # [0,1] -> [0,255] uint8
       
    print("Image generated successfully")
        
    return img


def create_sample_image(device, image_size=(64, 64), channels=3):
    """
    Create a sample image when no model is available.
    Useful for testing and development.
    
    Args:
        device: Device (cuda or cpu)
        image_size: Tuple (height, width)
        channels: Number of channels
        
    Returns:
        PIL.Image: Sample image
    """
    print("Creating sample image (development mode)...")
    
    # Create a colorful gradient pattern
    h, w = image_size
    
    if channels == 3:
        # Create RGB gradients
        r = np.linspace(0, 1, w).reshape(1, -1)
        r = np.repeat(r, h, axis=0)
        
        g = np.linspace(0, 1, h).reshape(-1, 1)
        g = np.repeat(g, w, axis=1)
        
        b = np.ones((h, w)) * 0.5
        
        # Combine channels
        image_array = np.stack([r, g, b], axis=2)
    else:
        # Grayscale
        image_array = np.random.random((h, w))
    
    # Convert to PIL image
    image_array = (image_array * 255).astype(np.uint8)
    
    if channels == 3:
        image = Image.fromarray(image_array, mode='RGB')
    else:
        image = Image.fromarray(image_array, mode='L')
    
    print("Sample image created")
    
    return image

