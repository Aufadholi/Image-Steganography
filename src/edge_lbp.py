"""
Edge Detection and Local Binary Pattern (LBP) Feature Extraction Module
Implements multiple edge detection algorithms and LBP for texture analysis
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import local_binary_pattern
from skimage.filters import sobel, prewitt


def preprocess_image(image):
    """
    Preprocess the input image: convert to grayscale and normalize
    
    Args:
        image: Input image (BGR or grayscale)
    
    Returns:
        Preprocessed grayscale image
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Normalize to 0-255 range
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """
    Apply Canny edge detection
    
    Args:
        image: Input grayscale image
        low_threshold: Lower threshold for edge linking
        high_threshold: Upper threshold for edge linking
    
    Returns:
        Binary edge map
    """
    return cv2.Canny(image, low_threshold, high_threshold)


def sobel_edge_detection(image):
    """
    Apply Sobel edge detection
    
    Args:
        image: Input grayscale image
    
    Returns:
        Edge magnitude map
    """
    # Calculate gradients in X and Y directions
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate edge magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    return magnitude.astype(np.uint8)


def prewitt_edge_detection(image):
    """
    Apply Prewitt edge detection
    
    Args:
        image: Input grayscale image
    
    Returns:
        Edge magnitude map
    """
    # Prewitt kernels
    kernel_x = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]])
    
    kernel_y = np.array([[-1, -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]])
    
    # Apply convolution
    grad_x = cv2.filter2D(image.astype(np.float32), -1, kernel_x)
    grad_y = cv2.filter2D(image.astype(np.float32), -1, kernel_y)
    
    # Calculate magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    return magnitude.astype(np.uint8)


def kirsch_edge_detection(image):
    """
    Apply Kirsch edge detection (8-directional)
    
    Args:
        image: Input grayscale image
    
    Returns:
        Edge magnitude map
    """
    # Kirsch kernels for 8 directions
    kernels = [
        np.array([[-3, -3, 5],
                 [-3, 0, 5],
                 [-3, -3, 5]]),
        np.array([[-3, 5, 5],
                 [-3, 0, 5],
                 [-3, -3, -3]]),
        np.array([[5, 5, 5],
                 [-3, 0, -3],
                 [-3, -3, -3]]),
        np.array([[5, 5, -3],
                 [5, 0, -3],
                 [-3, -3, -3]]),
        np.array([[5, -3, -3],
                 [5, 0, -3],
                 [5, -3, -3]]),
        np.array([[-3, -3, -3],
                 [5, 0, -3],
                 [5, 5, -3]]),
        np.array([[-3, -3, -3],
                 [-3, 0, -3],
                 [5, 5, 5]]),
        np.array([[-3, -3, -3],
                 [-3, 0, 5],
                 [-3, 5, 5]])
    ]
    
    responses = []
    for kernel in kernels:
        response = cv2.filter2D(image.astype(np.float32), -1, kernel)
        responses.append(response)
    
    # Take maximum response across all directions
    magnitude = np.maximum.reduce(responses)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    return magnitude.astype(np.uint8)


def laplacian_of_gaussian(image, sigma=1.0):
    """
    Apply Laplacian of Gaussian (LoG) edge detection
    
    Args:
        image: Input grayscale image
        sigma: Standard deviation for Gaussian blur
    
    Returns:
        Edge magnitude map
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # Apply Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Take absolute value and normalize
    magnitude = np.abs(laplacian)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    return magnitude.astype(np.uint8)


def multi_edge_detection(image, edge_threshold=0.3):
    """
    Apply multiple edge detection algorithms and combine results
    
    Args:
        image: Input grayscale image
        edge_threshold: Threshold for combining edge maps (0-1)
    
    Returns:
        Combined edge map and individual edge maps dictionary
    """
    # Apply all edge detection methods
    canny_edges = canny_edge_detection(image)
    sobel_edges = sobel_edge_detection(image)
    prewitt_edges = prewitt_edge_detection(image)
    kirsch_edges = kirsch_edge_detection(image)
    log_edges = laplacian_of_gaussian(image)
    
    # Normalize all edge maps to 0-1 range
    edge_maps = {
        'canny': canny_edges / 255.0,
        'sobel': sobel_edges / 255.0,
        'prewitt': prewitt_edges / 255.0,
        'kirsch': kirsch_edges / 255.0,
        'log': log_edges / 255.0
    }
    
    # Combine edge maps using weighted average
    weights = {'canny': 0.3, 'sobel': 0.2, 'prewitt': 0.2, 'kirsch': 0.15, 'log': 0.15}
    combined_edges = np.zeros_like(canny_edges, dtype=np.float32)
    
    for method, edge_map in edge_maps.items():
        combined_edges += weights[method] * edge_map
    
    # Apply threshold
    edge_mask = (combined_edges > edge_threshold).astype(np.uint8)
    
    # Convert back to 8-bit for consistency
    for method in edge_maps:
        edge_maps[method] = (edge_maps[method] * 255).astype(np.uint8)
    
    combined_edges = (combined_edges * 255).astype(np.uint8)
    
    return edge_mask, edge_maps, combined_edges


def extract_lbp_features(image, radius=1, n_points=8, method='uniform'):
    """
    Extract Local Binary Pattern (LBP) features
    
    Args:
        image: Input grayscale image
        radius: Radius of sample points
        n_points: Number of sample points
        method: LBP method ('uniform', 'default', 'ror', 'var')
    
    Returns:
        LBP image and texture strength map
    """
    # Compute LBP
    lbp = local_binary_pattern(image, n_points, radius, method=method)
    
    # Calculate texture strength using LBP variance
    # Higher variance indicates more texture
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    lbp_mean = cv2.filter2D(lbp, -1, kernel)
    lbp_variance = cv2.filter2D((lbp - lbp_mean) ** 2, -1, kernel)
    
    # Normalize texture strength
    texture_strength = cv2.normalize(lbp_variance, None, 0, 255, cv2.NORM_MINMAX)
    
    return lbp.astype(np.uint8), texture_strength.astype(np.uint8)


def select_embedding_pixels(image, edge_threshold=0.3, texture_threshold=0.4, 
                          max_capacity_ratio=0.1):
    """
    Select pixels for embedding based on edge and texture analysis
    
    Args:
        image: Input grayscale image
        edge_threshold: Threshold for edge detection
        texture_threshold: Threshold for texture strength
        max_capacity_ratio: Maximum ratio of pixels to use for embedding
    
    Returns:
        Binary mask of selected pixels, edge map, texture map, and embedding coordinates
    """
    # Get edge information
    edge_mask, edge_maps, combined_edges = multi_edge_detection(image, edge_threshold)
    
    # Get texture information
    lbp, texture_strength = extract_lbp_features(image)
    
    # Normalize texture strength to 0-1 range
    texture_normalized = texture_strength / 255.0
    
    # Create texture mask
    texture_mask = (texture_normalized > texture_threshold).astype(np.uint8)
    
    # Combine edge and texture masks
    # Prefer pixels that have both edge and texture characteristics
    combined_mask = np.logical_and(edge_mask, texture_mask).astype(np.uint8)
    
    # If not enough pixels, relax constraints
    total_pixels = image.shape[0] * image.shape[1]
    max_embedding_pixels = int(total_pixels * max_capacity_ratio)
    
    if np.sum(combined_mask) < max_embedding_pixels * 0.5:
        # Use edge OR texture if AND gives too few pixels
        combined_mask = np.logical_or(edge_mask, texture_mask).astype(np.uint8)
    
    # Get coordinates of selected pixels
    embedding_coords = np.where(combined_mask == 1)
    embedding_coords = list(zip(embedding_coords[0], embedding_coords[1]))
    
    # Limit to maximum capacity if needed
    if len(embedding_coords) > max_embedding_pixels:
        # Sort by combined edge + texture strength for better selection
        scores = []
        for y, x in embedding_coords:
            edge_score = combined_edges[y, x] / 255.0
            texture_score = texture_strength[y, x] / 255.0
            combined_score = edge_score + texture_score
            scores.append(combined_score)
        
        # Select top pixels by score
        sorted_indices = np.argsort(scores)[::-1][:max_embedding_pixels]
        embedding_coords = [embedding_coords[i] for i in sorted_indices]
        
        # Update mask
        combined_mask = np.zeros_like(combined_mask)
        for y, x in embedding_coords:
            combined_mask[y, x] = 1
    
    return combined_mask, edge_maps, texture_strength, embedding_coords


def adaptive_threshold_optimization(image, target_psnr=40.0, min_capacity_ratio=0.05):
    """
    Adaptively optimize thresholds for edge and texture detection to achieve target PSNR
    
    Args:
        image: Input grayscale image
        target_psnr: Target PSNR value
        min_capacity_ratio: Minimum capacity ratio to maintain
    
    Returns:
        Optimized edge_threshold, texture_threshold, and capacity_ratio
    """
    best_edge_threshold = 0.3
    best_texture_threshold = 0.4
    best_capacity_ratio = 0.1
    
    # Grid search for optimal thresholds
    edge_thresholds = np.arange(0.2, 0.6, 0.1)
    texture_thresholds = np.arange(0.3, 0.7, 0.1)
    capacity_ratios = np.arange(0.05, 0.15, 0.02)
    
    best_score = float('inf')
    
    for edge_th in edge_thresholds:
        for texture_th in texture_thresholds:
            for cap_ratio in capacity_ratios:
                if cap_ratio < min_capacity_ratio:
                    continue
                
                try:
                    mask, _, _, coords = select_embedding_pixels(
                        image, edge_th, texture_th, cap_ratio
                    )
                    
                    # Calculate a score based on number of selected pixels and distribution
                    num_pixels = len(coords)
                    if num_pixels == 0:
                        continue
                    
                    # Prefer configurations that provide good capacity while maintaining quality
                    capacity_score = num_pixels / (image.shape[0] * image.shape[1])
                    distribution_score = np.std([y for y, x in coords]) + np.std([x for y, x in coords])
                    
                    # Combined score (lower is better)
                    score = abs(capacity_score - cap_ratio) + 0.001 / (distribution_score + 1e-6)
                    
                    if score < best_score:
                        best_score = score
                        best_edge_threshold = edge_th
                        best_texture_threshold = texture_th
                        best_capacity_ratio = cap_ratio
                        
                except Exception:
                    continue
    
    return best_edge_threshold, best_texture_threshold, best_capacity_ratio


def adaptive_pixel_selection(image, payload_size, base_config, use_optimization=True):
    """
    Enhanced pixel selection with adaptive optimization capability
    
    Args:
        image: Input image
        payload_size: Size of payload in bytes
        base_config: Base configuration parameters
        use_optimization: Whether to use adaptive optimization
    
    Returns:
        Selected pixels, edge maps, texture map, and optimization info
    """
    # Preprocess image
    gray = preprocess_image(image)
    
    if use_optimization:
        try:
            # Import adaptive optimizer
            from .adaptive_optimizer import AdaptiveHybridSteganography
            
            # Initialize adaptive optimizer
            adaptive_optimizer = AdaptiveHybridSteganography()
            
            # Optimize parameters
            optimized_config = adaptive_optimizer.optimize_parameters(
                image, payload_size, base_config
            )
            
            # Use optimized parameters
            edge_threshold = optimized_config.get('edge_threshold', base_config.get('edge_threshold', 0.3))
            texture_threshold = optimized_config.get('texture_threshold', base_config.get('texture_threshold', 0.4))
            max_capacity_ratio = optimized_config.get('max_capacity_ratio', base_config.get('max_capacity_ratio', 0.1))
            
            print(f"📊 Using optimized parameters:")
            print(f"   Edge threshold: {edge_threshold:.3f}")
            print(f"   Texture threshold: {texture_threshold:.3f}")
            print(f"   Max capacity ratio: {max_capacity_ratio:.3f}")
            
        except Exception as e:
            print(f"⚠️  Adaptive optimization failed: {e}")
            print("   Falling back to base configuration...")
            use_optimization = False
    
    if not use_optimization:
        # Use base configuration
        edge_threshold = base_config.get('edge_threshold', 0.3)
        texture_threshold = base_config.get('texture_threshold', 0.4)
        max_capacity_ratio = base_config.get('max_capacity_ratio', 0.1)
    
    # Perform pixel selection with determined parameters
    mask, edge_maps, texture_strength, embedding_coords = select_embedding_pixels(
        gray, edge_threshold, texture_threshold, max_capacity_ratio
    )
    
    # Prepare optimization info
    optimization_info = {
        'adaptive_optimization_used': use_optimization,
        'final_edge_threshold': edge_threshold,
        'final_texture_threshold': texture_threshold,
        'final_capacity_ratio': max_capacity_ratio,
        'selected_pixels_count': len(embedding_coords),
        'total_pixels': gray.shape[0] * gray.shape[1],
        'utilization_ratio': len(embedding_coords) / (gray.shape[0] * gray.shape[1])
    }
    
    return mask, edge_maps, texture_strength, embedding_coords, optimization_info