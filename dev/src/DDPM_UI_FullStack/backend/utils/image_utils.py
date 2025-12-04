"""
Image Utilities
===============
Utilities for image processing and conversion.

Functions:
    - tensor_to_pil: Converts PyTorch tensor to PIL image
    - pil_to_base64: Converts PIL image to base64 string
    - normalize_tensor: Normalizes tensor to [0, 1] range
"""

# Import local modules
from src.config.libraries import *


def pil_to_base64(image, format='PNG'):
    """
    Converts a PIL image to base64 string.
    
    Args:
        image (PIL.Image): PIL image
        format (str): Output format ('PNG', 'JPEG', etc.)
    
    Returns:
        str: Base64 string of the image
    """
    # Create in-memory buffer
    buffered = BytesIO()
    
    # Save image to buffer
    image.save(buffered, format=format)
    
    # Obtener bytes
    img_bytes = buffered.getvalue()
    
    # Codificar a base64
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return img_base64
