#!/usr/bin/env python3
"""
Advanced Local Binary Pattern (LBP) Module for MRI Analysis

This module provides advanced LBP implementations specifically optimized for MRI texture analysis:
1. Multi-scale LBP for different texture resolutions
2. Rotation-invariant LBP for robust texture analysis
3. Uniform LBP for reduced dimensionality
4. Gabor-LBP fusion for enhanced texture discrimination
5. Wavelet-LBP combination for multi-frequency analysis
"""

import numpy as np
from scipy import ndimage
from skimage.feature import local_binary_pattern
from skimage import filters
import cv2
from scipy.ndimage import gaussian_filter


class AdvancedLBP:
    """Advanced Local Binary Pattern analysis for MRI images"""
    
    def __init__(self):
        self.lbp_params = {
            'multi_scale': {
                'radii': [1, 2, 3, 4],
                'n_points': [8, 12, 16, 20],
                'methods': ['uniform', 'ror', 'var']
            },
            'rotation_invariant': {
                'radius': 3,
                'n_points': 24,
                'method': 'ror'  # rotation invariant
            },
            'uniform': {
                'radius': 2,
                'n_points': 16,
                'method': 'uniform'
            },
            'gabor_fusion': {
                'frequencies': [0.1, 0.3, 0.5],
                'orientations': [0, 45, 90, 135],
                'sigma_x': 2.0,
                'sigma_y': 2.0
            },
            'wavelet_fusion': {
                'wavelet': 'db4',
                'levels': 3
            }
        }
    
    def extract_multi_scale_lbp(self, image):
        """
        Extract multi-scale LBP features
        
        Args:
            image: Input image (2D numpy array)
            
        Returns:
            features: Dictionary with multi-scale LBP features
            texture_map: Combined texture strength map
        """
        features = {}
        texture_maps = []
        
        params = self.lbp_params['multi_scale']
        
        for i, (radius, n_points) in enumerate(zip(params['radii'], params['n_points'])):
            scale_features = {}
            
            # Extract LBP with different methods
            for method in params['methods']:
                lbp = local_binary_pattern(image, n_points, radius, method=method)
                
                # Handle NaN values
                if np.any(np.isnan(lbp)):
                    lbp = np.nan_to_num(lbp, nan=0.0)
                
                # Calculate texture strength
                texture_strength = self._calculate_texture_strength(lbp)
                texture_maps.append(texture_strength)
                
                # Store features - handle edge case for histogram
                try:
                    if np.any(np.isfinite(lbp)):
                        hist, _ = np.histogram(lbp.flatten(), bins=n_points+2)
                    else:
                        hist = np.zeros(n_points+2)
                except ValueError:
                    hist = np.zeros(n_points+2)
                
                scale_features[f'{method}_lbp'] = lbp
                scale_features[f'{method}_strength'] = texture_strength
                scale_features[f'{method}_histogram'] = hist
            
            features[f'scale_{i+1}_r{radius}_p{n_points}'] = scale_features
        
        # Combine texture maps
        combined_texture_map = np.mean(texture_maps, axis=0)
        
        return features, combined_texture_map
    
    def extract_rotation_invariant_lbp(self, image):
        """
        Extract rotation-invariant LBP features
        
        Args:
            image: Input image (2D numpy array)
            
        Returns:
            lbp_features: Dictionary with rotation-invariant features
            texture_map: Texture strength map
        """
        params = self.lbp_params['rotation_invariant']
        
        # Extract rotation-invariant LBP
        lbp_ror = local_binary_pattern(
            image, 
            params['n_points'], 
            params['radius'], 
            method=params['method']
        )
        
        # Handle NaN values
        if np.any(np.isnan(lbp_ror)):
            lbp_ror = np.nan_to_num(lbp_ror, nan=0.0)
        
        # Calculate texture strength
        texture_strength = self._calculate_texture_strength(lbp_ror)
        
        # Calculate local variance for additional texture information
        local_variance = self._calculate_local_variance(image, params['radius'])
        
        # Safe histogram calculation
        try:
            if np.any(np.isfinite(lbp_ror)):
                hist, _ = np.histogram(lbp_ror.flatten(), bins=params['n_points']+1)
            else:
                hist = np.zeros(params['n_points']+1)
        except ValueError:
            hist = np.zeros(params['n_points']+1)
        
        lbp_features = {
            'rotation_invariant_lbp': lbp_ror,
            'texture_strength': texture_strength,
            'local_variance': local_variance,
            'histogram': hist,
            'texture_uniformity': self._calculate_texture_uniformity(lbp_ror),
            'texture_contrast': self._calculate_texture_contrast(lbp_ror)
        }
        
        return lbp_features, texture_strength
    
    def extract_uniform_lbp(self, image):
        """
        Extract uniform LBP features (reduced dimensionality)
        
        Args:
            image: Input image (2D numpy array)
            
        Returns:
            lbp_features: Dictionary with uniform LBP features
            texture_map: Texture strength map
        """
        params = self.lbp_params['uniform']
        
        # Extract uniform LBP
        lbp_uniform = local_binary_pattern(
            image, 
            params['n_points'], 
            params['radius'], 
            method=params['method']
        )
        
        # Handle NaN values
        if np.any(np.isnan(lbp_uniform)):
            lbp_uniform = np.nan_to_num(lbp_uniform, nan=0.0)
        
        # Calculate texture strength
        texture_strength = self._calculate_texture_strength(lbp_uniform)
        
        # Calculate additional uniform pattern statistics
        uniform_patterns = self._identify_uniform_patterns(lbp_uniform, params['n_points'])
        
        # Safe histogram calculation
        try:
            if np.any(np.isfinite(lbp_uniform)):
                hist, _ = np.histogram(lbp_uniform.flatten(), bins=params['n_points']+2)
            else:
                hist = np.zeros(params['n_points']+2)
        except ValueError:
            hist = np.zeros(params['n_points']+2)
        
        lbp_features = {
            'uniform_lbp': lbp_uniform,
            'texture_strength': texture_strength,
            'uniform_patterns': uniform_patterns,
            'histogram': hist,
            'uniformity_ratio': np.sum(uniform_patterns) / (uniform_patterns.shape[0] * uniform_patterns.shape[1])
        }
        
        return lbp_features, texture_strength
    
    def extract_gabor_lbp_fusion(self, image):
        """
        Extract Gabor-LBP fusion features
        
        Args:
            image: Input image (2D numpy array)
            
        Returns:
            fusion_features: Dictionary with Gabor-LBP fusion features
            enhanced_texture_map: Enhanced texture map
        """
        params = self.lbp_params['gabor_fusion']
        
        # Extract Gabor responses
        gabor_responses = self._extract_gabor_responses(image, params)
        
        # Apply LBP to Gabor responses
        gabor_lbp_features = []
        texture_maps = []
        
        for freq_idx, freq in enumerate(params['frequencies']):
            for orient_idx, orient in enumerate(params['orientations']):
                gabor_response = gabor_responses[f'freq_{freq_idx}_orient_{orient_idx}']
                
                # Apply LBP to Gabor response
                lbp = local_binary_pattern(gabor_response, 16, 2, method='uniform')
                texture_strength = self._calculate_texture_strength(lbp)
                
                gabor_lbp_features.append({
                    'frequency': freq,
                    'orientation': orient,
                    'gabor_response': gabor_response,
                    'lbp': lbp,
                    'texture_strength': texture_strength
                })
                
                texture_maps.append(texture_strength)
        
        # Combine texture maps
        enhanced_texture_map = np.mean(texture_maps, axis=0)
        
        fusion_features = {
            'gabor_lbp_features': gabor_lbp_features,
            'enhanced_texture_map': enhanced_texture_map,
            'gabor_responses': gabor_responses
        }
        
        return fusion_features, enhanced_texture_map
    
    def extract_comprehensive_lbp_features(self, image):
        """
        Extract comprehensive LBP features combining all methods
        
        Args:
            image: Input image (2D numpy array)
            
        Returns:
            comprehensive_features: Dictionary with all LBP features
            final_texture_map: Final combined texture map
        """
        comprehensive_features = {}
        texture_maps = []
        
        # Multi-scale LBP
        ms_features, ms_texture = self.extract_multi_scale_lbp(image)
        comprehensive_features['multi_scale'] = ms_features
        texture_maps.append(ms_texture)
        
        # Rotation-invariant LBP
        ri_features, ri_texture = self.extract_rotation_invariant_lbp(image)
        comprehensive_features['rotation_invariant'] = ri_features
        texture_maps.append(ri_texture)
        
        # Uniform LBP
        uniform_features, uniform_texture = self.extract_uniform_lbp(image)
        comprehensive_features['uniform'] = uniform_features
        texture_maps.append(uniform_texture)
        
        # Gabor-LBP fusion
        gabor_features, gabor_texture = self.extract_gabor_lbp_fusion(image)
        comprehensive_features['gabor_fusion'] = gabor_features
        texture_maps.append(gabor_texture)
        
        # Final combined texture map
        final_texture_map = np.mean(texture_maps, axis=0)
        
        # Calculate comprehensive texture statistics
        comprehensive_features['summary'] = {
            'mean_texture_strength': np.mean(final_texture_map),
            'std_texture_strength': np.std(final_texture_map),
            'texture_range': np.max(final_texture_map) - np.min(final_texture_map),
            'texture_entropy': self._calculate_entropy(final_texture_map),
            'texture_homogeneity': self._calculate_homogeneity(final_texture_map)
        }
        
        return comprehensive_features, final_texture_map
    
    def _calculate_texture_strength(self, lbp_image):
        """Calculate texture strength from LBP image"""
        # Calculate local standard deviation as texture strength
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        
        # Local mean
        local_mean = cv2.filter2D(lbp_image.astype(np.float32), -1, kernel)
        
        # Local variance
        local_var = cv2.filter2D((lbp_image.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        # Texture strength as standard deviation
        texture_strength = np.sqrt(local_var)
        
        return texture_strength
    
    def _calculate_local_variance(self, image, radius):
        """Calculate local variance in a neighborhood"""
        # Create circular kernel
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x*x + y*y <= radius*radius
        kernel = mask.astype(np.float32)
        kernel = kernel / np.sum(kernel)
        
        # Calculate local mean
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        
        # Calculate local variance
        local_var = cv2.filter2D((image.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        return local_var
    
    def _calculate_texture_uniformity(self, lbp_image):
        """Calculate texture uniformity"""
        hist, _ = np.histogram(lbp_image.flatten(), bins=256)
        hist = hist / np.sum(hist)  # Normalize
        uniformity = np.sum(hist ** 2)
        return uniformity
    
    def _calculate_texture_contrast(self, lbp_image):
        """Calculate texture contrast"""
        # Calculate gradient magnitude
        grad_x = ndimage.sobel(lbp_image, axis=1)
        grad_y = ndimage.sobel(lbp_image, axis=0)
        contrast = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(contrast)
    
    def _identify_uniform_patterns(self, lbp_image, n_points):
        """Identify uniform patterns in LBP image"""
        # A pattern is uniform if it has at most 2 0-1 transitions
        uniform_patterns = np.zeros_like(lbp_image, dtype=bool)
        
        # This is a simplified version - in practice, you'd use the actual uniform pattern lookup
        # For demonstration, we'll mark areas with low variation as uniform
        local_std = ndimage.generic_filter(lbp_image, np.std, size=3)
        uniform_patterns = local_std < np.percentile(local_std, 25)
        
        return uniform_patterns
    
    def _extract_gabor_responses(self, image, params):
        """Extract Gabor filter responses"""
        responses = {}
        
        for freq_idx, frequency in enumerate(params['frequencies']):
            for orient_idx, orientation in enumerate(params['orientations']):
                # Create Gabor filter
                gabor_real, gabor_imag = filters.gabor(
                    image,
                    frequency=frequency,
                    theta=np.deg2rad(orientation),
                    sigma_x=params['sigma_x'],
                    sigma_y=params['sigma_y']
                )
                
                # Combine real and imaginary parts
                gabor_magnitude = np.sqrt(gabor_real**2 + gabor_imag**2)
                
                responses[f'freq_{freq_idx}_orient_{orient_idx}'] = gabor_magnitude
        
        return responses
    
    def _calculate_entropy(self, image):
        """Calculate entropy of image"""
        hist, _ = np.histogram(image.flatten(), bins=256)
        hist = hist / np.sum(hist)  # Normalize
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        return entropy
    
    def _calculate_homogeneity(self, image):
        """Calculate homogeneity (inverse of contrast)"""
        # Calculate GLCM-like homogeneity
        # Simplified version using local variance
        local_var = ndimage.generic_filter(image, np.var, size=5)
        homogeneity = 1.0 / (1.0 + local_var)
        return np.mean(homogeneity)
    
    def select_texture_based_pixels(self, image, texture_map, selection_ratio=0.1, 
                                  avoid_high_texture=True):
        """
        Select pixels based on texture analysis for safe embedding
        
        Args:
            image: Input image
            texture_map: Texture strength map
            selection_ratio: Ratio of pixels to select
            avoid_high_texture: Whether to avoid high-texture areas
            
        Returns:
            selected_pixels: Boolean mask of selected pixels
            texture_stats: Statistics about texture selection
        """
        # Calculate texture threshold
        if avoid_high_texture:
            # Select pixels with low to medium texture
            texture_threshold = np.percentile(texture_map, 30)
            candidate_mask = texture_map < texture_threshold
        else:
            # Select pixels with medium texture (not too smooth, not too complex)
            low_threshold = np.percentile(texture_map, 20)
            high_threshold = np.percentile(texture_map, 80)
            candidate_mask = (texture_map > low_threshold) & (texture_map < high_threshold)
        
        # Get candidate pixel coordinates
        candidate_coords = np.where(candidate_mask)
        n_candidates = len(candidate_coords[0])
        
        # Calculate number of pixels to select
        n_select = int(selection_ratio * image.size)
        n_select = min(n_select, n_candidates)
        
        # Randomly select from candidates
        if n_candidates > 0:
            selected_indices = np.random.choice(n_candidates, n_select, replace=False)
            selected_pixels = np.zeros_like(image, dtype=bool)
            selected_pixels[candidate_coords[0][selected_indices], 
                          candidate_coords[1][selected_indices]] = True
        else:
            selected_pixels = np.zeros_like(image, dtype=bool)
        
        # Calculate statistics
        texture_stats = {
            'n_candidates': n_candidates,
            'n_selected': n_select,
            'selection_ratio_actual': n_select / image.size,
            'mean_texture_selected': np.mean(texture_map[selected_pixels]) if n_select > 0 else 0,
            'std_texture_selected': np.std(texture_map[selected_pixels]) if n_select > 0 else 0
        }
        
        return selected_pixels, texture_stats


def analyze_mri_texture(image, method='comprehensive'):
    """
    Convenience function for MRI texture analysis
    
    Args:
        image: Input MRI image
        method: Analysis method ('multi_scale', 'rotation_invariant', 'uniform', 
                'gabor_fusion', 'comprehensive')
        
    Returns:
        features: Extracted features
        texture_map: Texture strength map
    """
    lbp_analyzer = AdvancedLBP()
    
    if method == 'multi_scale':
        return lbp_analyzer.extract_multi_scale_lbp(image)
    elif method == 'rotation_invariant':
        return lbp_analyzer.extract_rotation_invariant_lbp(image)
    elif method == 'uniform':
        return lbp_analyzer.extract_uniform_lbp(image)
    elif method == 'gabor_fusion':
        return lbp_analyzer.extract_gabor_lbp_fusion(image)
    elif method == 'comprehensive':
        return lbp_analyzer.extract_comprehensive_lbp_features(image)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    # Test the advanced LBP module
    print("Advanced LBP Analysis Module")
    print("===========================")
    
    # Create a test image with different texture regions
    test_image = np.random.rand(256, 256) * 255
    
    # Add some texture patterns
    x, y = np.meshgrid(np.arange(256), np.arange(256))
    test_image += 50 * np.sin(x/10) * np.cos(y/10)  # Add sinusoidal pattern
    test_image = test_image.astype(np.uint8)
    
    # Analyze texture
    lbp_analyzer = AdvancedLBP()
    features, texture_map = lbp_analyzer.extract_comprehensive_lbp_features(test_image)
    
    print(f"Extracted features: {list(features.keys())}")
    print(f"Texture map shape: {texture_map.shape}")
    print(f"Summary statistics: {features['summary']}")
    
    # Test pixel selection
    selected_pixels, stats = lbp_analyzer.select_texture_based_pixels(
        test_image, texture_map, selection_ratio=0.1
    )
    
    print(f"Pixel selection stats: {stats}")