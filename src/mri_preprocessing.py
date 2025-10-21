#!/usr/bin/env python3
"""
MRI-Specific Preprocessing Module

This module provides preprocessing functions specifically designed for MRI images:
1. Denoising using Non-Local Means and BM3D-like methods
2. Intensity normalization for consistent contrast
3. Bias field correction for intensity inhomogeneity
4. Histogram equalization for enhanced contrast
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import filters, restoration, exposure
from skimage.filters import rank
from skimage.morphology import disk


class MRIPreprocessor:
    """MRI-specific preprocessing pipeline"""
    
    def __init__(self):
        self.preprocessing_params = {
            'denoising': {
                'method': 'non_local_means',
                'h': 10,
                'template_window_size': 7,
                'search_window_size': 21
            },
            'normalization': {
                'method': 'z_score',
                'clip_percentile': 99.5
            },
            'bias_correction': {
                'enabled': True,
                'iterations': 50,
                'convergence_threshold': 0.001
            },
            'contrast_enhancement': {
                'method': 'clahe',
                'clip_limit': 0.01,
                'tile_grid_size': (8, 8)
            }
        }
    
    def preprocess_mri(self, image, enable_denoising=True, enable_normalization=True, 
                       enable_bias_correction=True, enable_contrast_enhancement=True):
        """
        Complete MRI preprocessing pipeline
        
        Args:
            image: Input MRI image (numpy array)
            enable_denoising: Whether to apply denoising
            enable_normalization: Whether to apply intensity normalization
            enable_bias_correction: Whether to apply bias field correction
            enable_contrast_enhancement: Whether to apply contrast enhancement
            
        Returns:
            preprocessed_image: Processed MRI image
            preprocessing_info: Dictionary with processing information
        """
        preprocessed_image = image.copy()
        preprocessing_info = {
            'original_shape': image.shape,
            'original_dtype': image.dtype,
            'steps_applied': []
        }
        
        # Convert to float for processing
        if preprocessed_image.dtype != np.float64:
            preprocessed_image = preprocessed_image.astype(np.float64)
        
        # Step 1: Denoising
        if enable_denoising:
            preprocessed_image = self._apply_denoising(preprocessed_image)
            preprocessing_info['steps_applied'].append('denoising')
        
        # Step 2: Bias field correction
        if enable_bias_correction:
            preprocessed_image = self._apply_bias_correction(preprocessed_image)
            preprocessing_info['steps_applied'].append('bias_correction')
        
        # Step 3: Intensity normalization
        if enable_normalization:
            preprocessed_image = self._apply_normalization(preprocessed_image)
            preprocessing_info['steps_applied'].append('normalization')
        
        # Step 4: Contrast enhancement
        if enable_contrast_enhancement:
            preprocessed_image = self._apply_contrast_enhancement(preprocessed_image)
            preprocessing_info['steps_applied'].append('contrast_enhancement')
        
        # Convert back to original dtype range
        preprocessed_image = self._convert_to_original_range(preprocessed_image, image.dtype)
        preprocessing_info['final_shape'] = preprocessed_image.shape
        preprocessing_info['final_dtype'] = preprocessed_image.dtype
        
        return preprocessed_image, preprocessing_info
    
    def _apply_denoising(self, image):
        """Apply denoising using Non-Local Means"""
        params = self.preprocessing_params['denoising']
        
        if len(image.shape) == 3:
            # Color image
            denoised = np.zeros_like(image)
            for i in range(image.shape[2]):
                channel = image[:, :, i]
                # Normalize to 0-255 for cv2.fastNlMeansDenoising
                channel_norm = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype(np.uint8)
                denoised_channel = cv2.fastNlMeansDenoising(
                    channel_norm,
                    h=params['h'],
                    templateWindowSize=params['template_window_size'],
                    searchWindowSize=params['search_window_size']
                )
                # Convert back to original range
                denoised[:, :, i] = (denoised_channel.astype(np.float64) / 255.0) * (channel.max() - channel.min()) + channel.min()
        else:
            # Grayscale image
            # Normalize to 0-255 for cv2.fastNlMeansDenoising
            image_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            denoised_norm = cv2.fastNlMeansDenoising(
                image_norm,
                h=params['h'],
                templateWindowSize=params['template_window_size'],
                searchWindowSize=params['search_window_size']
            )
            # Convert back to original range
            denoised = (denoised_norm.astype(np.float64) / 255.0) * (image.max() - image.min()) + image.min()
        
        return denoised
    
    def _apply_bias_correction(self, image):
        """Apply bias field correction using N4 ITK-like method"""
        params = self.preprocessing_params['bias_correction']
        
        # Simple bias field correction using polynomial fitting
        # This is a simplified version of N4 bias correction
        
        if len(image.shape) == 3:
            corrected = np.zeros_like(image)
            for i in range(image.shape[2]):
                corrected[:, :, i] = self._correct_bias_field_2d(image[:, :, i], params)
        else:
            corrected = self._correct_bias_field_2d(image, params)
        
        return corrected
    
    def _correct_bias_field_2d(self, image_2d, params):
        """Correct bias field for 2D image"""
        # Create coordinate grids
        h, w = image_2d.shape
        y, x = np.mgrid[0:h, 0:w]
        
        # Normalize coordinates
        x_norm = (x - w/2) / (w/2)
        y_norm = (y - h/2) / (h/2)
        
        # Estimate bias field using polynomial fitting
        # Use a low-frequency polynomial to model the bias field
        valid_mask = image_2d > np.percentile(image_2d, 10)  # Exclude background
        
        if np.sum(valid_mask) < 100:  # Not enough valid pixels
            return image_2d
        
        # Create polynomial features
        features = np.column_stack([
            x_norm[valid_mask].flatten(),
            y_norm[valid_mask].flatten(),
            (x_norm[valid_mask]**2).flatten(),
            (y_norm[valid_mask]**2).flatten(),
            (x_norm[valid_mask] * y_norm[valid_mask]).flatten()
        ])
        
        intensities = image_2d[valid_mask].flatten()
        
        try:
            # Fit polynomial to estimate bias field
            coeffs = np.linalg.lstsq(features, intensities, rcond=None)[0]
            
            # Reconstruct bias field
            bias_field = (coeffs[0] * x_norm + 
                         coeffs[1] * y_norm + 
                         coeffs[2] * x_norm**2 + 
                         coeffs[3] * y_norm**2 + 
                         coeffs[4] * x_norm * y_norm)
            
            # Smooth the bias field
            bias_field = ndimage.gaussian_filter(bias_field, sigma=2.0)
            
            # Correct the image
            bias_field_normalized = bias_field / np.mean(bias_field[valid_mask])
            corrected = image_2d / (bias_field_normalized + 1e-8)
            
        except np.linalg.LinAlgError:
            # If fitting fails, return original image
            corrected = image_2d
        
        return corrected
    
    def _apply_normalization(self, image):
        """Apply intensity normalization"""
        params = self.preprocessing_params['normalization']
        
        if params['method'] == 'z_score':
            # Z-score normalization with clipping
            clip_val = np.percentile(image, params['clip_percentile'])
            clipped_image = np.clip(image, 0, clip_val)
            
            mean_val = np.mean(clipped_image)
            std_val = np.std(clipped_image)
            
            if std_val > 0:
                normalized = (image - mean_val) / std_val
            else:
                normalized = image - mean_val
                
        elif params['method'] == 'min_max':
            # Min-max normalization
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                normalized = (image - min_val) / (max_val - min_val)
            else:
                normalized = image - min_val
                
        else:
            normalized = image
        
        return normalized
    
    def _apply_contrast_enhancement(self, image):
        """Apply contrast enhancement using CLAHE"""
        params = self.preprocessing_params['contrast_enhancement']
        
        if params['method'] == 'clahe':
            if len(image.shape) == 3:
                enhanced = np.zeros_like(image)
                for i in range(image.shape[2]):
                    channel = image[:, :, i]
                    # Normalize to 0-1 for skimage
                    channel_norm = (channel - channel.min()) / (channel.max() - channel.min())
                    enhanced_channel = exposure.equalize_adapthist(
                        channel_norm,
                        clip_limit=params['clip_limit'],
                        nbins=256
                    )
                    # Convert back to original range
                    enhanced[:, :, i] = enhanced_channel * (channel.max() - channel.min()) + channel.min()
            else:
                # Normalize to 0-1 for skimage
                image_norm = (image - image.min()) / (image.max() - image.min())
                enhanced_norm = exposure.equalize_adapthist(
                    image_norm,
                    clip_limit=params['clip_limit'],
                    nbins=256
                )
                # Convert back to original range
                enhanced = enhanced_norm * (image.max() - image.min()) + image.min()
        else:
            enhanced = image
        
        return enhanced
    
    def _convert_to_original_range(self, processed_image, original_dtype):
        """Convert processed image back to original data type range"""
        if original_dtype == np.uint8:
            # For uint8, clip to 0-255
            processed_image = np.clip(processed_image, 0, 255)
            return processed_image.astype(np.uint8)
        elif original_dtype == np.uint16:
            # For uint16, clip to 0-65535
            processed_image = np.clip(processed_image, 0, 65535)
            return processed_image.astype(np.uint16)
        else:
            # For float types, keep as is
            return processed_image.astype(original_dtype)
    
    def get_preprocessing_quality_metrics(self, original_image, preprocessed_image):
        """Calculate quality metrics for preprocessing"""
        metrics = {}
        
        # Signal-to-noise ratio improvement
        original_std = np.std(original_image)
        preprocessed_std = np.std(preprocessed_image)
        
        metrics['noise_reduction_ratio'] = original_std / (preprocessed_std + 1e-8)
        
        # Contrast improvement
        original_contrast = np.std(original_image) / (np.mean(original_image) + 1e-8)
        preprocessed_contrast = np.std(preprocessed_image) / (np.mean(preprocessed_image) + 1e-8)
        
        metrics['contrast_improvement'] = preprocessed_contrast / (original_contrast + 1e-8)
        
        # Dynamic range utilization
        original_range = np.max(original_image) - np.min(original_image)
        preprocessed_range = np.max(preprocessed_image) - np.min(preprocessed_image)
        
        metrics['dynamic_range_ratio'] = preprocessed_range / (original_range + 1e-8)
        
        return metrics


def preprocess_mri_image(image, config=None):
    """
    Convenience function for MRI preprocessing
    
    Args:
        image: Input MRI image
        config: Preprocessing configuration (optional)
        
    Returns:
        preprocessed_image: Processed image
        info: Preprocessing information
    """
    preprocessor = MRIPreprocessor()
    
    if config:
        preprocessor.preprocessing_params.update(config)
    
    return preprocessor.preprocess_mri(image)


if __name__ == "__main__":
    # Test the preprocessing module
    print("MRI Preprocessing Module")
    print("========================")
    
    # Create a test image
    test_image = np.random.rand(256, 256) * 255
    test_image = test_image.astype(np.uint8)
    
    # Add some noise and bias
    noise = np.random.normal(0, 10, test_image.shape)
    test_image = test_image + noise
    
    # Preprocess
    preprocessor = MRIPreprocessor()
    processed, info = preprocessor.preprocess_mri(test_image)
    
    print(f"Original shape: {info['original_shape']}")
    print(f"Steps applied: {info['steps_applied']}")
    print(f"Final shape: {info['final_shape']}")
    
    # Calculate quality metrics
    metrics = preprocessor.get_preprocessing_quality_metrics(test_image, processed)
    print(f"Quality metrics: {metrics}")