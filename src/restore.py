"""
Restore Module for 100% Reversible Steganography
Implements perfect restoration of original cover image using trace matrix
"""

import numpy as np
import cv2
from .utils import load_trace_matrix


def restore_cover(stego_image, trace_matrix_path=None, trace_matrix=None):
    """
    Restore the original cover image from stego image using trace matrix
    
    Args:
        stego_image: Stego image as numpy array
        trace_matrix_path: Path to trace matrix file
        trace_matrix: Trace matrix dictionary (if not loading from file)
    
    Returns:
        Restored cover image (100% identical to original)
    """
    # Load trace matrix if not provided
    if trace_matrix is None:
        if trace_matrix_path is None:
            raise ValueError("Either trace_matrix or trace_matrix_path must be provided")
        trace_matrix = load_trace_matrix(trace_matrix_path)
    
    # Validate trace matrix
    required_keys = ['image_shape', 'embedding_coords', 'original_lsbs']
    for key in required_keys:
        if key not in trace_matrix:
            raise ValueError(f"Invalid trace matrix: missing key '{key}'")
    
    # Get restoration information
    embedding_coords = trace_matrix['embedding_coords']
    original_lsbs = trace_matrix['original_lsbs']
    original_shape = trace_matrix['image_shape']
    
    # Validate dimensions
    if stego_image.shape != original_shape:
        raise ValueError(f"Stego image shape {stego_image.shape} does not match "
                        f"expected shape {original_shape}")
    
    # Validate coordinate and LSB counts
    if len(embedding_coords) != len(original_lsbs):
        raise ValueError(f"Mismatch between embedding coordinates ({len(embedding_coords)}) "
                        f"and original LSBs ({len(original_lsbs)})")
    
    # Create a copy of stego image for restoration
    restored_image = stego_image.copy()
    
    # Determine which channel was modified during embedding
    if len(stego_image.shape) == 3:
        # For color images, restore the blue channel (channel 0)
        channel_to_restore = restored_image[:, :, 0]
    else:
        # For grayscale images
        channel_to_restore = restored_image
    
    # Restore original LSBs
    for i, ((y, x), original_lsb) in enumerate(zip(embedding_coords, original_lsbs)):
        # Validate coordinates
        if y >= channel_to_restore.shape[0] or x >= channel_to_restore.shape[1]:
            raise ValueError(f"Invalid coordinates ({y}, {x}) for image shape "
                           f"{channel_to_restore.shape}")
        
        # Clear current LSB and set original LSB
        current_pixel = channel_to_restore[y, x]
        restored_pixel = (current_pixel & 0xFE) | original_lsb
        channel_to_restore[y, x] = restored_pixel
    
    # Update the restored channel back to the image
    if len(stego_image.shape) == 3:
        restored_image[:, :, 0] = channel_to_restore
    else:
        restored_image = channel_to_restore
    
    return restored_image


def verify_restoration(original_cover, restored_cover):
    """
    Verify that restoration is 100% perfect
    
    Args:
        original_cover: Original cover image
        restored_cover: Restored cover image
    
    Returns:
        Verification results dictionary
    """
    # Check shape compatibility
    if original_cover.shape != restored_cover.shape:
        return {
            'perfect_restoration': False,
            'error': f"Shape mismatch: {original_cover.shape} vs {restored_cover.shape}",
            'mse': float('inf'),
            'max_difference': float('inf'),
            'identical_pixels': 0,
            'total_pixels': original_cover.size
        }
    
    # Calculate pixel-wise differences
    difference = np.abs(original_cover.astype(np.int32) - restored_cover.astype(np.int32))
    
    # Calculate metrics
    mse = np.mean(difference ** 2)
    max_difference = np.max(difference)
    identical_pixels = np.sum(difference == 0)
    total_pixels = original_cover.size
    
    # Perfect restoration means zero difference
    perfect_restoration = (mse == 0.0) and (max_difference == 0)
    
    verification_results = {
        'perfect_restoration': perfect_restoration,
        'mse': float(mse),
        'max_difference': int(max_difference),
        'identical_pixels': int(identical_pixels),
        'total_pixels': int(total_pixels),
        'identity_ratio': float(identical_pixels) / total_pixels,
        'different_pixels': int(total_pixels - identical_pixels)
    }
    
    if not perfect_restoration:
        # Find locations of different pixels
        diff_coords = np.where(difference > 0)
        if len(diff_coords[0]) > 0:
            verification_results['first_difference_location'] = (
                int(diff_coords[0][0]), int(diff_coords[1][0])
            )
            verification_results['first_difference_values'] = (
                int(original_cover[diff_coords[0][0], diff_coords[1][0]]),
                int(restored_cover[diff_coords[0][0], diff_coords[1][0]])
            )
    
    return verification_results


def batch_restore(stego_images, trace_matrices, output_dir):
    """
    Batch restore multiple stego images
    
    Args:
        stego_images: List of stego image paths or arrays
        trace_matrices: List of trace matrix paths or dictionaries
        output_dir: Output directory for restored images
    
    Returns:
        List of restoration results
    """
    import os
    from .utils import load_image, save_image
    
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for i, (stego, trace) in enumerate(zip(stego_images, trace_matrices)):
        try:
            # Load stego image if it's a path
            if isinstance(stego, str):
                stego_img = load_image(stego)
                stego_name = os.path.splitext(os.path.basename(stego))[0]
            else:
                stego_img = stego
                stego_name = f"stego_{i}"
            
            # Load trace matrix if it's a path
            if isinstance(trace, str):
                trace_matrix = load_trace_matrix(trace)
            else:
                trace_matrix = trace
            
            # Restore cover image
            restored_image = restore_cover(stego_img, trace_matrix=trace_matrix)
            
            # Save restored image
            restored_path = os.path.join(output_dir, f"{stego_name}_restored.png")
            save_image(restored_image, restored_path)
            
            result = {
                'stego_name': stego_name,
                'restored_path': restored_path,
                'success': True
            }
            
        except Exception as e:
            result = {
                'stego_name': stego_name if 'stego_name' in locals() else f"stego_{i}",
                'error': str(e),
                'success': False
            }
        
        results.append(result)
    
    return results


def create_restoration_report(original_cover, stego_image, restored_cover, trace_matrix):
    """
    Create a comprehensive restoration report
    
    Args:
        original_cover: Original cover image
        stego_image: Stego image
        restored_cover: Restored cover image
        trace_matrix: Trace matrix used for restoration
    
    Returns:
        Detailed restoration report dictionary
    """
    # Verify restoration
    verification = verify_restoration(original_cover, restored_cover)
    
    # Calculate embedding impact
    embedding_difference = np.abs(original_cover.astype(np.int32) - stego_image.astype(np.int32))
    embedding_mse = np.mean(embedding_difference ** 2)
    embedding_max_diff = np.max(embedding_difference)
    modified_pixels = np.sum(embedding_difference > 0)
    
    # Get trace matrix information
    num_embedded_pixels = len(trace_matrix.get('embedding_coords', []))
    
    report = {
        'restoration_verification': verification,
        'embedding_impact': {
            'mse_original_to_stego': float(embedding_mse),
            'max_difference_original_to_stego': int(embedding_max_diff),
            'modified_pixels': int(modified_pixels),
            'total_pixels': int(original_cover.size),
            'modification_ratio': float(modified_pixels) / original_cover.size
        },
        'trace_matrix_info': {
            'embedded_pixels': num_embedded_pixels,
            'has_payload_metadata': 'payload_metadata' in trace_matrix,
            'has_config': 'config' in trace_matrix,
            'image_shape': trace_matrix.get('image_shape', 'unknown')
        },
        'reversibility_guarantee': {
            'lsb_restoration_possible': True,
            'data_integrity_preserved': verification['perfect_restoration'],
            'no_information_loss': verification['mse'] == 0.0
        }
    }
    
    return report


def validate_trace_matrix(trace_matrix, stego_image_shape):
    """
    Validate trace matrix for compatibility with stego image
    
    Args:
        trace_matrix: Trace matrix dictionary
        stego_image_shape: Shape of the stego image
    
    Returns:
        Validation results dictionary
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required fields
    required_fields = ['image_shape', 'embedding_coords', 'original_lsbs']
    for field in required_fields:
        if field not in trace_matrix:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Missing required field: {field}")
    
    if not validation_results['valid']:
        return validation_results
    
    # Check image shape compatibility
    expected_shape = trace_matrix['image_shape']
    if expected_shape != stego_image_shape:
        validation_results['valid'] = False
        validation_results['errors'].append(
            f"Image shape mismatch: expected {expected_shape}, got {stego_image_shape}"
        )
    
    # Check coordinate and LSB count consistency
    coords_count = len(trace_matrix['embedding_coords'])
    lsbs_count = len(trace_matrix['original_lsbs'])
    if coords_count != lsbs_count:
        validation_results['valid'] = False
        validation_results['errors'].append(
            f"Coordinate count ({coords_count}) != LSB count ({lsbs_count})"
        )
    
    # Check coordinate bounds
    if validation_results['valid']:
        max_y, max_x = stego_image_shape[:2]
        for i, (y, x) in enumerate(trace_matrix['embedding_coords']):
            if y < 0 or y >= max_y or x < 0 or x >= max_x:
                validation_results['valid'] = False
                validation_results['errors'].append(
                    f"Invalid coordinate at index {i}: ({y}, {x}) out of bounds for shape {stego_image_shape}"
                )
                break
    
    # Check LSB values
    if validation_results['valid']:
        for i, lsb in enumerate(trace_matrix['original_lsbs']):
            if lsb not in [0, 1]:
                validation_results['valid'] = False
                validation_results['errors'].append(
                    f"Invalid LSB value at index {i}: {lsb} (should be 0 or 1)"
                )
                break
    
    # Optional field warnings
    optional_fields = ['payload_metadata', 'config', 'payload_length']
    for field in optional_fields:
        if field not in trace_matrix:
            validation_results['warnings'].append(f"Optional field missing: {field}")
    
    # Performance warnings
    if validation_results['valid']:
        embedding_ratio = coords_count / (stego_image_shape[0] * stego_image_shape[1])
        if embedding_ratio > 0.1:
            validation_results['warnings'].append(
                f"High embedding ratio: {embedding_ratio:.2%} (>10%)"
            )
    
    return validation_results


def secure_restore(stego_image, trace_matrix_path, password=None):
    """
    Secure restoration with optional password protection
    
    Args:
        stego_image: Stego image
        trace_matrix_path: Path to trace matrix
        password: Optional password for trace matrix decryption
    
    Returns:
        Restored cover image
    """
    # For basic implementation, password protection is not implemented
    # This would require encryption/decryption of trace matrix
    if password is not None:
        raise NotImplementedError("Password protection not implemented in this version")
    
    return restore_cover(stego_image, trace_matrix_path)


def incremental_restore(stego_image, trace_matrix, partial_coords=None):
    """
    Incrementally restore parts of the image (useful for debugging)
    
    Args:
        stego_image: Stego image
        trace_matrix: Trace matrix
        partial_coords: List of coordinate indices to restore (None for all)
    
    Returns:
        Partially restored image
    """
    if partial_coords is None:
        return restore_cover(stego_image, trace_matrix=trace_matrix)
    
    # Create copy for partial restoration
    partially_restored = stego_image.copy()
    
    # Get restoration data
    embedding_coords = trace_matrix['embedding_coords']
    original_lsbs = trace_matrix['original_lsbs']
    
    # Determine channel to restore
    if len(stego_image.shape) == 3:
        channel_to_restore = partially_restored[:, :, 0]
    else:
        channel_to_restore = partially_restored
    
    # Restore only specified coordinates
    for idx in partial_coords:
        if 0 <= idx < len(embedding_coords):
            y, x = embedding_coords[idx]
            original_lsb = original_lsbs[idx]
            
            current_pixel = channel_to_restore[y, x]
            restored_pixel = (current_pixel & 0xFE) | original_lsb
            channel_to_restore[y, x] = restored_pixel
    
    # Update channel back to image
    if len(stego_image.shape) == 3:
        partially_restored[:, :, 0] = channel_to_restore
    else:
        partially_restored = channel_to_restore
    
    return partially_restored