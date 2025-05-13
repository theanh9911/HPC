# API Image Preprocessing Module
# api/preprocess.py

import numpy as np
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_image(image_data, target_size=(224, 224)):
    """
    Preprocess an image for model inference.
    
    Args:
        image_data (bytes): Raw image data
        target_size (tuple): Target size for the image (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed image ready for model input
    """
    try:
        # Open image from binary data
        img = Image.open(io.BytesIO(image_data))
        
        # Convert grayscale to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image to target size
        img = img.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Image preprocessed successfully to shape {img_array.shape}")
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise ValueError(f"Error preprocessing image: {str(e)}")

def apply_image_enhancement(image_data, enhance_contrast=True, denoise=False):
    """
    Apply image enhancements to improve X-ray image quality.
    
    Args:
        image_data (bytes): Raw image data
        enhance_contrast (bool): Whether to enhance contrast
        denoise (bool): Whether to apply denoising
        
    Returns:
        bytes: Enhanced image data
    """
    try:
        # Open image from binary data
        img = Image.open(io.BytesIO(image_data))
        
        # Convert grayscale to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply enhancements if requested
        if enhance_contrast:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)  # Enhance contrast by factor of 1.5
            logger.info("Contrast enhancement applied")
        
        if denoise:
            # Simple denoising by slight blur
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
            logger.info("Denoising applied")
        
        # Convert back to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        enhanced_data = buffer.getvalue()
        
        return enhanced_data
        
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        # Return original image if enhancement fails
        return image_data

def verify_image(image_data):
    """
    Verify if the uploaded file is a valid image and if it appears to be a chest X-ray.
    
    Args:
        image_data (bytes): Raw image data
        
    Returns:
        bool: Whether the image is valid and appears to be a chest X-ray
    """
    try:
        # Check if it's a valid image
        try:
            img = Image.open(io.BytesIO(image_data))
            img.verify()  # Verify it's an image
        except:
            logger.warning("Invalid image file uploaded")
            return False
        
        # Basic check for chest X-ray: Typically grayscale and certain dimensions
        img = Image.open(io.BytesIO(image_data))
        
        # Check if grayscale or close to grayscale
        if img.mode == 'RGB':
            # Convert to numpy and check if RGB channels are similar (grayscale-like)
            img_array = np.array(img)
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            rg_diff = np.abs(r - g).mean()
            rb_diff = np.abs(r - b).mean()
            gb_diff = np.abs(g - b).mean()
            
            if rg_diff > 10 or rb_diff > 10 or gb_diff > 10:
                logger.warning("Image doesn't appear to be grayscale (potentially not an X-ray)")
                # We'll still process it, just log a warning
        
        # Check image dimensions (chest X-rays tend to have certain aspect ratios)
        width, height = img.size
        aspect_ratio = width / height
        
        # Most chest X-rays are roughly square or slightly portrait/landscape
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            logger.warning(f"Unusual aspect ratio for chest X-ray: {aspect_ratio}")
            # We'll still process it, just log a warning
        
        return True
        
    except Exception as e:
        logger.error(f"Error verifying image: {e}")
        return False
