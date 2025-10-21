#!/usr/bin/env python3
"""
ROI-Adaptive Embedding Module for MRI Images

This module provides Region of Interest (ROI) adaptive embedding functionality:
1. Automatic segmentation of diagnostic regions in MRI
2. Safe area identification for steganographic embedding
3. Adaptive embedding that avoids critical diagnostic areas
4. Multiple segmentation methods (intensity-based, morphological, deep learning-like)
5. Quality assessment for diagnostic preservation
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import filters, morphology, measure, segmentation
from skimage.morphology import disk, ball, binary_erosion, binary_dilation
from skimage.feature import canny
import warnings


class ROIAdaptiveEmbedding:
    """ROI-Adaptive embedding for MRI images"""
    
    def __init__(self):
        self.segmentation_params = {
            'intensity_based': {
                'brain_threshold_percentile': 10,  # Threshold to separate brain from background
                'ventricle_threshold_percentile': 85,  # Threshold for ventricles (dark regions)
                'white_matter_threshold_percentile': 60,  # Threshold for white matter
                'lesion_detection_sensitivity': 0.8
            },
            'morphological': {
                'erosion_kernel_size': 3,
                'dilation_kernel_size': 5,
                'min_region_size': 100,
                'connectivity': 2
            },
            'edge_based': {
                'edge_threshold': 0.1,
                'min_contour_area': 50,
                'max_contour_area': 10000
            },
            'safety_margins': {
                'critical_region_buffer': 5,  # pixels to expand around critical regions
                'embedding_exclusion_ratio': 0.3,  # ratio of image to exclude from embedding
                'min_safe_distance': 3  # minimum distance from critical regions
            }
        }
        
        self.diagnostic_regions = {
            'brain_tissue': [],
            'ventricles': [],
            'white_matter': [],
            'gray_matter': [],
            'potential_lesions': [],
            'edges_boundaries': [],
            'background': []
        }
    
    def segment_mri_regions(self, image, segmentation_method='comprehensive'):
        """
        Segment MRI image into diagnostic and non-diagnostic regions
        
        Args:
            image: Input MRI image (2D numpy array)
            segmentation_method: Method to use ('intensity', 'morphological', 'edge', 'comprehensive')
            
        Returns:
            segmentation_result: Dictionary with segmented regions
            diagnostic_mask: Boolean mask of diagnostic regions to avoid
            safe_embedding_mask: Boolean mask of safe regions for embedding
        """
        segmentation_result = {}
        
        if segmentation_method == 'intensity' or segmentation_method == 'comprehensive':
            intensity_regions = self._segment_by_intensity(image)
            segmentation_result.update(intensity_regions)
        
        if segmentation_method == 'morphological' or segmentation_method == 'comprehensive':
            morphological_regions = self._segment_by_morphology(image)
            segmentation_result.update(morphological_regions)
        
        if segmentation_method == 'edge' or segmentation_method == 'comprehensive':
            edge_regions = self._segment_by_edges(image)
            segmentation_result.update(edge_regions)
        
        # Combine all diagnostic regions
        diagnostic_mask = self._combine_diagnostic_regions(segmentation_result)
        
        # Create safe embedding mask
        safe_embedding_mask = self._create_safe_embedding_mask(image, diagnostic_mask)
        
        # Store results
        self.diagnostic_regions = segmentation_result
        
        return segmentation_result, diagnostic_mask, safe_embedding_mask
    
    def _segment_by_intensity(self, image):
        """Segment regions based on intensity values"""
        params = self.segmentation_params['intensity_based']
        regions = {}
        
        # Calculate intensity thresholds
        brain_threshold = np.percentile(image, params['brain_threshold_percentile'])
        ventricle_threshold = np.percentile(image, params['ventricle_threshold_percentile'])
        white_matter_threshold = np.percentile(image, params['white_matter_threshold_percentile'])
        
        # Segment brain tissue (exclude background)
        brain_mask = image > brain_threshold
        regions['brain_tissue'] = brain_mask
        
        # Segment ventricles (dark regions within brain)
        ventricle_candidates = (image < ventricle_threshold) & brain_mask
        # Clean up small regions
        ventricle_mask = morphology.remove_small_objects(ventricle_candidates, min_size=50)
        regions['ventricles'] = ventricle_mask
        
        # Segment white matter (bright regions)
        white_matter_candidates = (image > white_matter_threshold) & brain_mask
        white_matter_mask = morphology.remove_small_objects(white_matter_candidates, min_size=100)
        regions['white_matter'] = white_matter_mask
        
        # Segment gray matter (medium intensity)
        gray_matter_mask = brain_mask & ~white_matter_mask & ~ventricle_mask
        regions['gray_matter'] = gray_matter_mask
        
        # Detect potential lesions (abnormal intensities)
        lesions_mask = self._detect_potential_lesions(image, brain_mask, params)
        regions['potential_lesions'] = lesions_mask
        
        # Background
        regions['background'] = ~brain_mask
        
        return regions
    
    def _segment_by_morphology(self, image):
        """Segment regions using morphological operations"""
        params = self.segmentation_params['morphological']
        regions = {}
        
        # Create binary image
        threshold = filters.threshold_otsu(image)
        binary = image > threshold
        
        # Morphological operations
        kernel = disk(params['erosion_kernel_size'])
        eroded = binary_erosion(binary, kernel)
        
        kernel = disk(params['dilation_kernel_size'])
        dilated = binary_dilation(eroded, kernel)
        
        # Remove small objects
        cleaned = morphology.remove_small_objects(
            dilated, 
            min_size=params['min_region_size'],
            connectivity=params['connectivity']
        )
        
        # Label connected components
        labeled = measure.label(cleaned, connectivity=params['connectivity'])
        
        # Analyze regions
        region_props = measure.regionprops(labeled)
        
        # Classify regions based on properties
        large_regions = np.zeros_like(image, dtype=bool)
        small_regions = np.zeros_like(image, dtype=bool)
        
        for prop in region_props:
            if prop.area > 1000:  # Large regions (likely main brain structures)
                large_regions[labeled == prop.label] = True
            else:  # Smaller regions
                small_regions[labeled == prop.label] = True
        
        regions['morphological_large'] = large_regions
        regions['morphological_small'] = small_regions
        
        return regions
    
    def _segment_by_edges(self, image):
        """Segment regions based on edge information"""
        params = self.segmentation_params['edge_based']
        regions = {}
        
        # Detect edges
        edges = canny(image, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Analyze contours
        edge_regions = np.zeros_like(image, dtype=bool)
        boundary_regions = np.zeros_like(image, dtype=bool)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if params['min_contour_area'] < area < params['max_contour_area']:
                # Create mask from contour
                mask = np.zeros_like(image, dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                
                # Classify based on area
                if area > 500:
                    boundary_regions |= (mask > 0)
                else:
                    edge_regions |= (mask > 0)
        
        regions['edges_boundaries'] = boundary_regions
        regions['edge_details'] = edge_regions
        
        return regions
    
    def _detect_potential_lesions(self, image, brain_mask, params):
        """Detect potential lesions or abnormalities"""
        # Calculate local statistics within brain
        brain_pixels = image[brain_mask]
        mean_intensity = np.mean(brain_pixels)
        std_intensity = np.std(brain_pixels)
        
        # Define abnormal intensity ranges
        abnormal_low = mean_intensity - 2 * std_intensity
        abnormal_high = mean_intensity + 2 * std_intensity
        
        # Find pixels with abnormal intensities
        lesion_candidates = brain_mask & (
            (image < abnormal_low) | (image > abnormal_high)
        )
        
        # Remove small artifacts
        lesions_mask = morphology.remove_small_objects(lesion_candidates, min_size=20)
        
        # Additional filtering based on local contrast
        if np.sum(lesions_mask) > 0:
            # Calculate local contrast around lesion candidates
            kernel = disk(3)
            local_std = ndimage.generic_filter(image.astype(np.float32), np.std, footprint=kernel)
            
            # Keep only lesions with significant local contrast
            high_contrast_mask = local_std > np.percentile(local_std[brain_mask], 70)
            lesions_mask = lesions_mask & high_contrast_mask
        
        return lesions_mask
    
    def _combine_diagnostic_regions(self, segmentation_result):
        """Combine all diagnostic regions into a single mask"""
        diagnostic_mask = np.zeros_like(list(segmentation_result.values())[0], dtype=bool)
        
        # Critical regions that should be avoided for embedding
        critical_regions = [
            'potential_lesions',
            'ventricles',
            'edges_boundaries',
            'morphological_large'
        ]
        
        for region_name in critical_regions:
            if region_name in segmentation_result:
                diagnostic_mask |= segmentation_result[region_name]
        
        # Add safety buffer around critical regions
        safety_params = self.segmentation_params['safety_margins']
        kernel = disk(safety_params['critical_region_buffer'])
        diagnostic_mask = binary_dilation(diagnostic_mask, kernel)
        
        return diagnostic_mask
    
    def _create_safe_embedding_mask(self, image, diagnostic_mask):
        """Create mask of safe regions for embedding"""
        safety_params = self.segmentation_params['safety_margins']
        
        # Start with entire image
        safe_mask = np.ones_like(image, dtype=bool)
        
        # Exclude diagnostic regions
        safe_mask = safe_mask & ~diagnostic_mask
        
        # Exclude background (very low intensity regions)
        background_threshold = np.percentile(image, 5)
        safe_mask = safe_mask & (image > background_threshold)
        
        # Exclude very high intensity regions (potential artifacts)
        artifact_threshold = np.percentile(image, 95)
        safe_mask = safe_mask & (image < artifact_threshold)
        
        # Calculate distance from diagnostic regions
        distance_map = ndimage.distance_transform_edt(~diagnostic_mask)
        safe_mask = safe_mask & (distance_map >= safety_params['min_safe_distance'])
        
        # Limit embedding area to prevent over-embedding
        total_pixels = np.sum(safe_mask)
        max_embedding_pixels = int(image.size * (1 - safety_params['embedding_exclusion_ratio']))
        
        if total_pixels > max_embedding_pixels:
            # Randomly select subset of safe pixels
            safe_coords = np.where(safe_mask)
            selected_indices = np.random.choice(
                len(safe_coords[0]), 
                max_embedding_pixels, 
                replace=False
            )
            
            # Create new mask with selected pixels
            new_safe_mask = np.zeros_like(safe_mask)
            new_safe_mask[safe_coords[0][selected_indices], safe_coords[1][selected_indices]] = True
            safe_mask = new_safe_mask
        
        return safe_mask
    
    def adaptive_pixel_selection(self, image, edge_map, texture_map, payload_size):
        """
        Select pixels adaptively based on ROI analysis, edge detection, and texture analysis
        
        Args:
            image: Input MRI image
            edge_map: Edge strength map
            texture_map: Texture strength map
            payload_size: Required number of pixels for embedding
            
        Returns:
            selected_pixels: Boolean mask of selected pixels
            selection_info: Information about the selection process
        """
        # Perform ROI segmentation
        segmentation_result, diagnostic_mask, safe_embedding_mask = self.segment_mri_regions(image)
        
        # Get safe pixels
        safe_coords = np.where(safe_embedding_mask)
        n_safe_pixels = len(safe_coords[0])
        
        if n_safe_pixels < payload_size:
            warnings.warn(f"Not enough safe pixels ({n_safe_pixels}) for payload size ({payload_size})")
            payload_size = n_safe_pixels
        
        if payload_size == 0 or n_safe_pixels == 0:
            selected_pixels = np.zeros_like(image, dtype=bool)
            selection_info = {
                'n_selected': 0,
                'n_safe_available': n_safe_pixels,
                'selection_method': 'none',
                'safety_ratio': 0,
                'mean_suitability_score': 0,
                'segmentation_summary': {
                    'n_diagnostic_pixels': np.sum(diagnostic_mask),
                    'n_safe_pixels': n_safe_pixels,
                    'diagnostic_ratio': np.sum(diagnostic_mask) / image.size
                }
            }
            return selected_pixels, selection_info
        
        # Calculate combined suitability score
        suitability_scores = np.zeros(n_safe_pixels)
        
        # Pre-calculate distance transform once (optimization)
        distance_map = ndimage.distance_transform_edt(~diagnostic_mask)
        max_distance = np.max(distance_map)
        
        for i, (y, x) in enumerate(zip(safe_coords[0], safe_coords[1])):
            # Lower edge strength is better (less likely to affect diagnosis)
            edge_score = 1.0 - (edge_map[y, x] / (np.max(edge_map) + 1e-8))
            
            # Medium texture strength is preferred
            texture_score = 1.0 - abs(texture_map[y, x] - np.median(texture_map)) / (np.max(texture_map) + 1e-8)
            
            # Distance from diagnostic regions (farther is better)
            distance_score = distance_map[y, x] / (max_distance + 1e-8)
            
            # Combined score
            suitability_scores[i] = (edge_score + texture_score + distance_score) / 3.0
        
        # Select pixels with highest suitability scores
        best_indices = np.argsort(suitability_scores)[-payload_size:]
        
        # Create selection mask
        selected_pixels = np.zeros_like(image, dtype=bool)
        selected_y = safe_coords[0][best_indices]
        selected_x = safe_coords[1][best_indices]
        selected_pixels[selected_y, selected_x] = True
        
        selection_info = {
            'n_selected': payload_size,
            'n_safe_available': n_safe_pixels,
            'selection_method': 'roi_adaptive',
            'safety_ratio': payload_size / n_safe_pixels if n_safe_pixels > 0 else 0,
            'mean_suitability_score': np.mean(suitability_scores[best_indices]),
            'segmentation_summary': {
                'n_diagnostic_pixels': np.sum(diagnostic_mask),
                'n_safe_pixels': n_safe_pixels,
                'diagnostic_ratio': np.sum(diagnostic_mask) / image.size
            }
        }
        
        return selected_pixels, selection_info
    
    def validate_embedding_safety(self, image, selected_pixels, diagnostic_mask):
        """
        Validate that embedding locations are safe for medical diagnosis
        
        Args:
            image: Original MRI image
            selected_pixels: Boolean mask of selected embedding pixels
            diagnostic_mask: Boolean mask of diagnostic regions
            
        Returns:
            safety_report: Dictionary with safety validation results
        """
        safety_report = {}
        
        # Check overlap with diagnostic regions
        overlap = selected_pixels & diagnostic_mask
        n_overlap = np.sum(overlap)
        
        safety_report['diagnostic_overlap'] = {
            'n_pixels': n_overlap,
            'percentage': n_overlap / np.sum(selected_pixels) * 100 if np.sum(selected_pixels) > 0 else 0,
            'is_safe': n_overlap == 0
        }
        
        # Check intensity distribution of selected pixels
        if np.sum(selected_pixels) > 0:
            selected_intensities = image[selected_pixels]
            overall_intensities = image.flatten()
            
            safety_report['intensity_analysis'] = {
                'mean_selected': np.mean(selected_intensities),
                'std_selected': np.std(selected_intensities),
                'mean_overall': np.mean(overall_intensities),
                'std_overall': np.std(overall_intensities),
                'intensity_range_ok': (
                    np.min(selected_intensities) > np.percentile(overall_intensities, 10) and
                    np.max(selected_intensities) < np.percentile(overall_intensities, 90)
                )
            }
        
        # Calculate minimum distance from diagnostic regions
        if np.sum(selected_pixels) > 0 and np.sum(diagnostic_mask) > 0:
            distance_map = ndimage.distance_transform_edt(~diagnostic_mask)
            min_distance = np.min(distance_map[selected_pixels])
            mean_distance = np.mean(distance_map[selected_pixels])
            
            safety_report['distance_analysis'] = {
                'min_distance_from_diagnostic': min_distance,
                'mean_distance_from_diagnostic': mean_distance,
                'safe_distance': min_distance >= self.segmentation_params['safety_margins']['min_safe_distance']
            }
        
        # Overall safety assessment
        is_safe = (
            safety_report['diagnostic_overlap']['is_safe'] and
            safety_report.get('intensity_analysis', {}).get('intensity_range_ok', True) and
            safety_report.get('distance_analysis', {}).get('safe_distance', True)
        )
        
        safety_report['overall_safety'] = {
            'is_safe': is_safe,
            'confidence': 'high' if is_safe else 'low',
            'recommendation': 'proceed' if is_safe else 'review_selection'
        }
        
        return safety_report
    
    def get_diagnostic_preservation_metrics(self, original_image, stego_image, diagnostic_mask):
        """
        Calculate metrics to assess preservation of diagnostic information
        
        Args:
            original_image: Original MRI image
            stego_image: Stego image after embedding
            diagnostic_mask: Boolean mask of diagnostic regions
            
        Returns:
            preservation_metrics: Dictionary with preservation metrics
        """
        preservation_metrics = {}
        
        if np.sum(diagnostic_mask) == 0:
            preservation_metrics['error'] = 'No diagnostic regions identified'
            return preservation_metrics
        
        # Extract diagnostic regions
        original_diagnostic = original_image[diagnostic_mask]
        stego_diagnostic = stego_image[diagnostic_mask]
        
        # Calculate differences in diagnostic regions
        diff = np.abs(original_diagnostic.astype(np.float32) - stego_diagnostic.astype(np.float32))
        
        preservation_metrics['diagnostic_region_analysis'] = {
            'mean_absolute_difference': np.mean(diff),
            'max_absolute_difference': np.max(diff),
            'std_difference': np.std(diff),
            'relative_error': np.mean(diff) / (np.mean(original_diagnostic) + 1e-8),
            'unchanged_pixels': np.sum(diff == 0),
            'changed_pixels': np.sum(diff > 0),
            'preservation_ratio': np.sum(diff == 0) / len(diff)
        }
        
        # Calculate structural similarity in diagnostic regions
        if len(original_diagnostic) > 1:
            correlation = np.corrcoef(original_diagnostic, stego_diagnostic)[0, 1]
            preservation_metrics['structural_similarity'] = {
                'correlation': correlation if not np.isnan(correlation) else 1.0,
                'is_well_preserved': correlation > 0.99 if not np.isnan(correlation) else True
            }
        
        return preservation_metrics


def perform_roi_adaptive_embedding(image, edge_map, texture_map, payload_size):
    """
    Convenience function for ROI-adaptive embedding
    
    Args:
        image: Input MRI image
        edge_map: Edge strength map
        texture_map: Texture strength map
        payload_size: Required number of pixels for embedding
        
    Returns:
        selected_pixels: Boolean mask of selected pixels
        roi_info: Information about ROI analysis and selection
    """
    roi_embedder = ROIAdaptiveEmbedding()
    selected_pixels, selection_info = roi_embedder.adaptive_pixel_selection(
        image, edge_map, texture_map, payload_size
    )
    
    # Validate safety
    _, diagnostic_mask, _ = roi_embedder.segment_mri_regions(image)
    safety_report = roi_embedder.validate_embedding_safety(image, selected_pixels, diagnostic_mask)
    
    roi_info = {
        'selection_info': selection_info,
        'safety_report': safety_report,
        'diagnostic_regions': roi_embedder.diagnostic_regions
    }
    
    return selected_pixels, roi_info


if __name__ == "__main__":
    # Test the ROI-adaptive embedding module
    print("ROI-Adaptive Embedding Module")
    print("=============================")
    
    # Create a test MRI-like image
    test_image = np.random.rand(256, 256) * 200 + 50
    
    # Add some brain-like structures
    center = (128, 128)
    y, x = np.ogrid[:256, :256]
    brain_mask = ((x - center[0])**2 + (y - center[1])**2) < 100**2
    test_image[brain_mask] += 100
    
    # Add some ventricle-like dark regions
    ventricle_centers = [(100, 128), (156, 128)]
    for vc in ventricle_centers:
        ventricle_mask = ((x - vc[0])**2 + (y - vc[1])**2) < 20**2
        test_image[ventricle_mask] -= 80
    
    test_image = test_image.astype(np.uint8)
    
    # Test ROI segmentation
    roi_embedder = ROIAdaptiveEmbedding()
    segmentation_result, diagnostic_mask, safe_mask = roi_embedder.segment_mri_regions(test_image)
    
    print(f"Segmentation regions: {list(segmentation_result.keys())}")
    print(f"Diagnostic pixels: {np.sum(diagnostic_mask)}")
    print(f"Safe pixels: {np.sum(safe_mask)}")
    
    # Test adaptive pixel selection
    edge_map = np.random.rand(256, 256)
    texture_map = np.random.rand(256, 256)
    
    selected_pixels, roi_info = perform_roi_adaptive_embedding(
        test_image, edge_map, texture_map, payload_size=1000
    )
    
    print(f"Selected pixels: {np.sum(selected_pixels)}")
    print(f"Selection info: {roi_info['selection_info']}")
    print(f"Safety assessment: {roi_info['safety_report']['overall_safety']}")