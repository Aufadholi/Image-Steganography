"""
Steganography Embedding and Extraction Module
Implements reversible LSB embedding using edge detection and LBP features
"""

import numpy as np
import cv2
import os
from .edge_lbp import select_embedding_pixels, preprocess_image
from .utils import (
    prepare_payload, add_payload_metadata_header, extract_payload_metadata_header,
    create_trace_matrix, save_trace_matrix, calculate_embedding_capacity,
    validate_payload_size, extract_payload_from_binary
)


def embed_payload(cover_image, payload_path, config=None):
    """
    Embed payload into cover image using hybrid edge detection + LBP + reversible LSB
    
    Args:
        cover_image: Cover image as numpy array
        payload_path: Path to payload file
        config: Configuration dictionary with embedding parameters
    
    Returns:
        Tuple of (stego_image, trace_matrix, embedding_info)
    """
    if config is None:
        config = {
            'edge_threshold': 0.3,
            'texture_threshold': 0.4,
            'max_capacity_ratio': 0.1,
            'use_adaptive_threshold': True,
            'add_metadata_header': True
        }
    
    # Preprocess cover image
    cover_gray = preprocess_image(cover_image)
    
    # Get payload size for optimization
    import os
    payload_size = os.path.getsize(payload_path) if os.path.exists(payload_path) else 0
    
    # Enhanced pixel selection with adaptive optimization
    use_adaptive = config.get('use_adaptive_optimization', False)
    
    if use_adaptive:
        print("🧠 Using adaptive multi-objective optimization...")
        from .edge_lbp import adaptive_pixel_selection
        
        # Use enhanced adaptive pixel selection
        embedding_mask, edge_maps, texture_strength, embedding_coords, optimization_info = adaptive_pixel_selection(
            cover_image, payload_size, config, use_optimization=True
        )
        
        # Update config with optimized parameters
        config.update({
            'edge_threshold': optimization_info['final_edge_threshold'],
            'texture_threshold': optimization_info['final_texture_threshold'],
            'max_capacity_ratio': optimization_info['final_capacity_ratio']
        })
        
    else:
        print("⚙️  Using standard pixel selection...")
        # Adaptive threshold optimization if enabled (legacy)
        if config.get('use_adaptive_threshold', True):
            from .edge_lbp import adaptive_threshold_optimization
            edge_th, texture_th, capacity_ratio = adaptive_threshold_optimization(
                cover_gray, target_psnr=config.get('target_psnr', 40.0)
            )
            config['edge_threshold'] = edge_th
            config['texture_threshold'] = texture_th
            config['max_capacity_ratio'] = capacity_ratio

        # Select embedding pixels using edge detection and LBP
        embedding_mask, edge_maps, texture_strength, embedding_coords = select_embedding_pixels(
            cover_gray, 
            config['edge_threshold'], 
            config['texture_threshold'],
            config['max_capacity_ratio']
        )
    
    # Calculate embedding capacity
    capacity = calculate_embedding_capacity(cover_gray.shape, embedding_coords)
    
    # Prepare payload
    payload_binary, payload_metadata = prepare_payload(payload_path)
    
    # Add metadata header if enabled
    if config.get('add_metadata_header', True):
        payload_binary = add_payload_metadata_header(payload_binary, payload_metadata)
    
    # Validate payload size
    if not validate_payload_size(payload_binary, capacity):
        raise ValueError(f"Payload too large. Required: {len(payload_binary)} bits, "
                        f"Available: {capacity['max_bits']} bits")
    
    # Pad payload to fit exactly in available space if needed
    if len(payload_binary) < capacity['max_bits']:
        padding_bits = capacity['max_bits'] - len(payload_binary)
        payload_binary += '0' * padding_bits
    
    # Create stego image (copy of cover)
    if len(cover_image.shape) == 3:
        stego_image = cover_image.copy()
        # For color images, embed in the blue channel (least perceptible)
        channel_to_modify = stego_image[:, :, 0]  # Blue channel
    else:
        stego_image = cover_image.copy()
        channel_to_modify = stego_image
    
    # Store original LSBs for reversibility
    original_lsbs = []
    
    # Embed payload using LSB
    for i, (y, x) in enumerate(embedding_coords):
        if i < len(payload_binary):
            # Store original LSB
            original_lsb = channel_to_modify[y, x] & 1
            original_lsbs.append(original_lsb)
            
            # Clear LSB and set new bit
            channel_to_modify[y, x] = (channel_to_modify[y, x] & 0xFE) | int(payload_binary[i])
        else:
            break
    
    # Update the modified channel back to stego image
    if len(cover_image.shape) == 3:
        stego_image[:, :, 0] = channel_to_modify
    else:
        stego_image = channel_to_modify
    
    # Create trace matrix for reversibility
    trace_matrix = create_trace_matrix(cover_image.shape, embedding_coords, original_lsbs)
    trace_matrix['payload_metadata'] = payload_metadata
    trace_matrix['config'] = config
    trace_matrix['payload_length'] = len(payload_binary)
    
    # Calculate PSNR for embedding quality
    psnr_value = calculate_psnr(cover_image, stego_image)
    
    # Embedding information
    embedding_info = {
        'capacity': capacity,
        'payload_size_bits': len(payload_binary),
        'payload_size_bytes': len(payload_binary) // 8,
        'utilization_ratio': len(payload_binary) / capacity['max_bits'],
        'embedding_coords_count': len(embedding_coords),
        'edge_threshold': config['edge_threshold'],
        'texture_threshold': config['texture_threshold'],
        'psnr': psnr_value,
        'config': config
    }
    
    return stego_image, trace_matrix, embedding_info


def extract_payload(stego_image, trace_matrix_path=None, trace_matrix=None, 
                   output_path=None, config=None):
    """
    Extract payload from stego image
    
    Args:
        stego_image: Stego image as numpy array
        trace_matrix_path: Path to trace matrix file
        trace_matrix: Trace matrix dictionary (if not loading from file)
        output_path: Path where to save extracted payload
        config: Configuration dictionary
    
    Returns:
        Extracted payload binary and metadata
    """
    if config is None:
        config = {}
    
    # Load trace matrix if not provided
    if trace_matrix is None:
        if trace_matrix_path is None:
            raise ValueError("Either trace_matrix or trace_matrix_path must be provided")
        from .utils import load_trace_matrix
        trace_matrix = load_trace_matrix(trace_matrix_path)
    
    # Get embedding information from trace matrix
    embedding_coords = trace_matrix['embedding_coords']
    payload_length = trace_matrix.get('payload_length', len(embedding_coords))
    payload_metadata = trace_matrix.get('payload_metadata', {})
    
    # Extract from appropriate channel
    if len(stego_image.shape) == 3:
        channel_to_extract = stego_image[:, :, 0]  # Blue channel
    else:
        channel_to_extract = stego_image
    
    # Extract LSBs - use the full payload_length from trace matrix
    extracted_bits = []
    for i, (y, x) in enumerate(embedding_coords):
        if i < payload_length:
            lsb = channel_to_extract[y, x] & 1
            extracted_bits.append(str(lsb))
        else:
            break
    
    # Convert to binary string
    payload_binary = ''.join(extracted_bits)
    
    # Remove padding by finding the actual payload length
    if trace_matrix.get('config', {}).get('add_metadata_header', True):
        try:
            # Extract metadata header to get actual payload length
            payload_binary_clean, extracted_metadata = extract_payload_metadata_header(payload_binary)
            payload_metadata.update(extracted_metadata)
        except Exception:
            # Fallback: use payload_binary as is
            payload_binary_clean = payload_binary
    else:
        payload_binary_clean = payload_binary
    
    # Save extracted payload if output path is provided
    if output_path is not None:
        extract_payload_from_binary(payload_binary_clean, payload_metadata, output_path)
    
    extraction_info = {
        'extracted_bits': len(payload_binary),
        'payload_bits': len(payload_binary_clean),
        'metadata': payload_metadata,
        'extraction_coords_count': len(embedding_coords)
    }
    
    return payload_binary_clean, payload_metadata, extraction_info


def embed_text(cover_image, text, config=None):
    """
    Convenience function to embed text directly
    
    Args:
        cover_image: Cover image as numpy array
        text: Text string to embed
        config: Configuration dictionary
    
    Returns:
        Tuple of (stego_image, trace_matrix, embedding_info)
    """
    # Create temporary text file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
        temp_file.write(text)
        temp_path = temp_file.name
    
    try:
        result = embed_payload(cover_image, temp_path, config)
        return result
    finally:
        # Clean up temporary file
        os.unlink(temp_path)


def extract_text(stego_image, trace_matrix_path=None, trace_matrix=None):
    """
    Convenience function to extract text directly
    
    Args:
        stego_image: Stego image as numpy array
        trace_matrix_path: Path to trace matrix file
        trace_matrix: Trace matrix dictionary
    
    Returns:
        Extracted text string
    """
    payload_binary, metadata, _ = extract_payload(
        stego_image, trace_matrix_path, trace_matrix
    )
    
    if metadata.get('type') == 'text':
        from .utils import binary_to_text
        return binary_to_text(payload_binary)
    else:
        raise ValueError("Payload is not text type")


def calculate_psnr(original, modified):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        original: Original image
        modified: Modified image
    
    Returns:
        PSNR value in dB
    """
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


def calculate_ssim(original, modified):
    """
    Calculate Structural Similarity Index (SSIM)
    
    Args:
        original: Original image
        modified: Modified image
    
    Returns:
        SSIM value
    """
    from skimage.metrics import structural_similarity as ssim
    
    # Convert to grayscale if needed
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        modified_gray = cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        modified_gray = modified
    
    ssim_value = ssim(original_gray, modified_gray)
    return ssim_value


def verify_embedding_quality(cover_image, stego_image, min_psnr=30.0, min_ssim=0.9):
    """
    Verify the quality of embedding by checking PSNR and SSIM thresholds
    
    Args:
        cover_image: Original cover image
        stego_image: Stego image
        min_psnr: Minimum acceptable PSNR
        min_ssim: Minimum acceptable SSIM
    
    Returns:
        Quality verification dictionary
    """
    psnr = calculate_psnr(cover_image, stego_image)
    ssim = calculate_ssim(cover_image, stego_image)
    
    quality_check = {
        'psnr': psnr,
        'ssim': ssim,
        'psnr_acceptable': psnr >= min_psnr,
        'ssim_acceptable': ssim >= min_ssim,
        'overall_quality': psnr >= min_psnr and ssim >= min_ssim
    }
    
    return quality_check


def adaptive_embedding(cover_image, payload_path, target_psnr=40.0, max_iterations=10):
    """
    Adaptively adjust embedding parameters to achieve target PSNR
    
    Args:
        cover_image: Cover image
        payload_path: Path to payload file
        target_psnr: Target PSNR value
        max_iterations: Maximum optimization iterations
    
    Returns:
        Best embedding result achieving target PSNR
    """
    best_result = None
    best_psnr = 0
    
    # Try different configurations
    edge_thresholds = np.linspace(0.2, 0.5, 5)
    texture_thresholds = np.linspace(0.3, 0.6, 5)
    
    for i, edge_th in enumerate(edge_thresholds):
        for j, texture_th in enumerate(texture_thresholds):
            if i * len(texture_thresholds) + j >= max_iterations:
                break
                
            config = {
                'edge_threshold': edge_th,
                'texture_threshold': texture_th,
                'max_capacity_ratio': 0.1,
                'use_adaptive_threshold': False,
                'add_metadata_header': True
            }
            
            try:
                stego_image, trace_matrix, embedding_info = embed_payload(
                    cover_image, payload_path, config
                )
                
                psnr = calculate_psnr(cover_image, stego_image)
                
                if psnr >= target_psnr and psnr > best_psnr:
                    best_psnr = psnr
                    best_result = (stego_image, trace_matrix, embedding_info)
                    
                    # If we achieved target PSNR, we can stop
                    if psnr >= target_psnr:
                        break
                        
            except Exception as e:
                # Skip this configuration if it fails
                continue
    
    if best_result is None:
        raise ValueError(f"Could not achieve target PSNR of {target_psnr}. "
                        "Try reducing target PSNR or using a smaller payload.")
    
    return best_result


def batch_embed(cover_images, payload_paths, output_dir, config=None):
    """
    Batch embed multiple payloads into multiple cover images
    
    Args:
        cover_images: List of cover image paths or arrays
        payload_paths: List of payload paths
        output_dir: Output directory for results
        config: Configuration dictionary
    
    Returns:
        List of embedding results
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for i, (cover, payload) in enumerate(zip(cover_images, payload_paths)):
        try:
            # Load cover image if it's a path
            if isinstance(cover, str):
                from .utils import load_image
                cover_img = load_image(cover)
                cover_name = os.path.splitext(os.path.basename(cover))[0]
            else:
                cover_img = cover
                cover_name = f"image_{i}"
            
            # Embed payload
            stego_image, trace_matrix, embedding_info = embed_payload(
                cover_img, payload, config
            )
            
            # Save results
            stego_path = os.path.join(output_dir, f"{cover_name}_stego.png")
            trace_path = os.path.join(output_dir, f"{cover_name}_trace.pkl")
            
            from .utils import save_image, save_trace_matrix
            save_image(stego_image, stego_path)
            save_trace_matrix(trace_matrix, trace_path)
            
            result = {
                'cover_name': cover_name,
                'stego_path': stego_path,
                'trace_path': trace_path,
                'embedding_info': embedding_info,
                'success': True
            }
            
        except Exception as e:
            result = {
                'cover_name': cover_name if 'cover_name' in locals() else f"image_{i}",
                'error': str(e),
                'success': False
            }
        
        results.append(result)
    
    return results