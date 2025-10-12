"""
Evaluation Module for Steganography Quality Assessment
Implements comprehensive metrics: PSNR, SSIM, entropy, histogram, correlation, reversibility
"""

import numpy as np
import cv2
import json
from datetime import datetime
from scipy import stats
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from .restore import verify_restoration


def calculate_psnr(original, modified):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        original: Original image
        modified: Modified image
    
    Returns:
        PSNR value in dB
    """
    try:
        # Handle different image types
        if len(original.shape) != len(modified.shape):
            raise ValueError("Image dimensions don't match")
        
        # Convert to same data type for comparison
        original = original.astype(np.float64)
        modified = modified.astype(np.float64)
        
        mse = np.mean((original - modified) ** 2)
        if mse == 0:
            return float('inf')
        
        max_pixel_value = 255.0
        psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return float(psnr_value)
        
    except Exception as e:
        return {'error': str(e)}


def calculate_ssim(original, modified):
    """
    Calculate Structural Similarity Index (SSIM)
    
    Args:
        original: Original image
        modified: Modified image
    
    Returns:
        SSIM value and detailed metrics
    """
    try:
        # Convert to grayscale if needed
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            modified_gray = cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original
            modified_gray = modified
        
        # Calculate SSIM with full return for detailed metrics
        ssim_value, ssim_image = ssim(
            original_gray, modified_gray, 
            full=True, 
            data_range=255
        )
        
        ssim_metrics = {
            'ssim': float(ssim_value),
            'mean_ssim': float(np.mean(ssim_image)),
            'std_ssim': float(np.std(ssim_image)),
            'min_ssim': float(np.min(ssim_image)),
            'max_ssim': float(np.max(ssim_image))
        }
        
        return ssim_metrics
        
    except Exception as e:
        return {'error': str(e)}


def calculate_entropy(image):
    """
    Calculate image entropy
    
    Args:
        image: Input image
    
    Returns:
        Entropy value and histogram
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate histogram
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        
        # Normalize histogram to get probabilities
        hist = hist / np.sum(hist)
        
        # Remove zero probabilities to avoid log(0)
        hist = hist[hist > 0]
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        return {
            'entropy': float(entropy),
            'max_entropy': 8.0,  # Maximum for 8-bit images
            'entropy_ratio': float(entropy / 8.0)
        }
        
    except Exception as e:
        return {'error': str(e)}


def calculate_histogram_similarity(original, modified):
    """
    Calculate histogram similarity between images
    
    Args:
        original: Original image
        modified: Modified image
    
    Returns:
        Various histogram similarity metrics
    """
    try:
        # Convert to grayscale if needed
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            mod_gray = cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original
            mod_gray = modified
        
        # Calculate histograms
        hist_orig = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
        hist_mod = cv2.calcHist([mod_gray], [0], None, [256], [0, 256])
        
        # Normalize histograms
        hist_orig = hist_orig / np.sum(hist_orig)
        hist_mod = hist_mod / np.sum(hist_mod)
        
        # Calculate various similarity metrics
        correlation = cv2.compareHist(hist_orig, hist_mod, cv2.HISTCMP_CORREL)
        chi_square = cv2.compareHist(hist_orig, hist_mod, cv2.HISTCMP_CHISQR)
        intersection = cv2.compareHist(hist_orig, hist_mod, cv2.HISTCMP_INTERSECT)
        bhattacharyya = cv2.compareHist(hist_orig, hist_mod, cv2.HISTCMP_BHATTACHARYYA)
        
        # Calculate Kullback-Leibler divergence
        kl_divergence = 0
        for i in range(len(hist_orig)):
            if hist_orig[i] > 0 and hist_mod[i] > 0:
                kl_divergence += hist_orig[i] * np.log(hist_orig[i] / hist_mod[i])
        
        return {
            'correlation': float(correlation),
            'chi_square': float(chi_square),
            'intersection': float(intersection),
            'bhattacharyya': float(bhattacharyya),
            'kl_divergence': float(kl_divergence)
        }
        
    except Exception as e:
        return {'error': str(e)}


def calculate_correlation(original, modified):
    """
    Calculate pixel-wise correlation between images
    
    Args:
        original: Original image
        modified: Modified image
    
    Returns:
        Correlation metrics
    """
    try:
        # Flatten images for correlation calculation
        orig_flat = original.flatten().astype(np.float64)
        mod_flat = modified.flatten().astype(np.float64)
        
        # Calculate Pearson correlation
        pearson_corr, pearson_p = stats.pearsonr(orig_flat, mod_flat)
        
        # Calculate Spearman correlation
        spearman_corr, spearman_p = stats.spearmanr(orig_flat, mod_flat)
        
        # Calculate normalized cross-correlation
        ncc = np.corrcoef(orig_flat, mod_flat)[0, 1]
        
        return {
            'pearson_correlation': float(pearson_corr),
            'pearson_p_value': float(pearson_p),
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p),
            'normalized_cross_correlation': float(ncc)
        }
        
    except Exception as e:
        return {'error': str(e)}


def calculate_advanced_metrics(original, modified):
    """
    Calculate advanced image quality metrics
    
    Args:
        original: Original image
        modified: Modified image
    
    Returns:
        Advanced metrics dictionary
    """
    try:
        # Convert to grayscale for some metrics
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            mod_gray = cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original
            mod_gray = modified
        
        # Mean Squared Error
        mse = np.mean((orig_gray.astype(np.float64) - mod_gray.astype(np.float64)) ** 2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(orig_gray.astype(np.float64) - mod_gray.astype(np.float64)))
        
        # Maximum Absolute Error
        max_ae = np.max(np.abs(orig_gray.astype(np.float64) - mod_gray.astype(np.float64)))
        
        # Signal-to-Noise Ratio
        signal_power = np.mean(orig_gray.astype(np.float64) ** 2)
        noise_power = mse
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # Normalized Absolute Error
        nae = np.sum(np.abs(orig_gray.astype(np.float64) - mod_gray.astype(np.float64))) / np.sum(orig_gray.astype(np.float64))
        
        # Universal Image Quality Index
        mu1 = np.mean(orig_gray)
        mu2 = np.mean(mod_gray)
        sigma1_sq = np.var(orig_gray)
        sigma2_sq = np.var(mod_gray)
        sigma12 = np.mean((orig_gray - mu1) * (mod_gray - mu2))
        
        uqi = (4 * sigma12 * mu1 * mu2) / ((sigma1_sq + sigma2_sq) * (mu1**2 + mu2**2))
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'max_absolute_error': float(max_ae),
            'snr': float(snr),
            'normalized_absolute_error': float(nae),
            'universal_quality_index': float(uqi)
        }
        
    except Exception as e:
        return {'error': str(e)}


def evaluate_reversibility(original_cover, stego_image, trace_matrix):
    """
    Evaluate reversibility of the steganography method
    
    Args:
        original_cover: Original cover image
        stego_image: Stego image
        trace_matrix: Trace matrix for restoration
    
    Returns:
        Reversibility evaluation results
    """
    try:
        from .restore import restore_cover, verify_restoration
        
        # Restore the cover image
        restored_cover = restore_cover(stego_image, trace_matrix=trace_matrix)
        
        # Verify restoration
        verification = verify_restoration(original_cover, restored_cover)
        
        # Additional reversibility metrics
        reversibility_metrics = {
            'restoration_verification': verification,
            'perfect_reversibility': verification['perfect_restoration'],
            'lossless_restoration': verification['mse'] == 0.0,
            'embedding_pixels': len(trace_matrix.get('embedding_coords', [])),
            'trace_matrix_size': len(str(trace_matrix)),  # Approximate size
        }
        
        return reversibility_metrics
        
    except Exception as e:
        return {'error': str(e)}


def evaluate_metrics(cover_image, stego_image, payload_path=None, trace_matrix=None):
    """
    Comprehensive evaluation of steganography quality
    
    Args:
        cover_image: Original cover image
        stego_image: Stego image
        payload_path: Path to payload file (optional)
        trace_matrix: Trace matrix for reversibility testing (optional)
    
    Returns:
        Complete evaluation report
    """
    evaluation_report = {
        'timestamp': datetime.now().isoformat(),
        'image_info': {
            'cover_shape': cover_image.shape,
            'stego_shape': stego_image.shape,
            'image_size': cover_image.size,
            'data_type': str(cover_image.dtype)
        }
    }
    
    # Basic quality metrics
    evaluation_report['psnr'] = calculate_psnr(cover_image, stego_image)
    evaluation_report['ssim'] = calculate_ssim(cover_image, stego_image)
    
    # Entropy analysis
    evaluation_report['entropy'] = {
        'cover': calculate_entropy(cover_image),
        'stego': calculate_entropy(stego_image)
    }
    
    # Histogram similarity
    evaluation_report['histogram_similarity'] = calculate_histogram_similarity(cover_image, stego_image)
    
    # Correlation analysis
    evaluation_report['correlation'] = calculate_correlation(cover_image, stego_image)
    
    # Advanced metrics
    evaluation_report['advanced_metrics'] = calculate_advanced_metrics(cover_image, stego_image)
    
    # Reversibility evaluation if trace matrix is provided
    if trace_matrix is not None:
        evaluation_report['reversibility'] = evaluate_reversibility(
            cover_image, stego_image, trace_matrix
        )
    
    # Payload information if available
    if payload_path is not None:
        from .utils import get_file_info
        payload_info = get_file_info(payload_path)
        if payload_info:
            evaluation_report['payload_info'] = payload_info
    
    # Overall quality assessment
    evaluation_report['quality_assessment'] = assess_overall_quality(evaluation_report)
    
    return evaluation_report


def assess_overall_quality(evaluation_report):
    """
    Assess overall quality based on multiple metrics
    
    Args:
        evaluation_report: Complete evaluation report
    
    Returns:
        Overall quality assessment
    """
    try:
        # Extract key metrics
        psnr_value = evaluation_report.get('psnr', 0)
        ssim_value = evaluation_report.get('ssim', {}).get('ssim', 0)
        correlation = evaluation_report.get('correlation', {}).get('pearson_correlation', 0)
        reversibility = evaluation_report.get('reversibility', {}).get('perfect_reversibility', False)
        
        # Define quality thresholds
        thresholds = {
            'excellent': {'psnr': 45, 'ssim': 0.98, 'correlation': 0.99},
            'good': {'psnr': 35, 'ssim': 0.95, 'correlation': 0.95},
            'acceptable': {'psnr': 25, 'ssim': 0.90, 'correlation': 0.90},
            'poor': {'psnr': 15, 'ssim': 0.80, 'correlation': 0.80}
        }
        
        # Assess quality level
        quality_level = 'poor'
        for level, thresh in thresholds.items():
            if (psnr_value >= thresh['psnr'] and 
                ssim_value >= thresh['ssim'] and 
                correlation >= thresh['correlation']):
                quality_level = level
                break
        
        # Calculate composite score
        psnr_score = min(psnr_value / 50.0, 1.0)  # Normalize to 0-1
        ssim_score = ssim_value
        corr_score = correlation
        
        composite_score = (psnr_score + ssim_score + corr_score) / 3.0
        
        assessment = {
            'quality_level': quality_level,
            'composite_score': float(composite_score),
            'imperceptibility_grade': 'A' if psnr_value >= 40 else 'B' if psnr_value >= 30 else 'C' if psnr_value >= 20 else 'D',
            'reversibility_status': 'Perfect' if reversibility else 'Unknown' if 'reversibility' not in evaluation_report else 'Failed',
            'recommendations': generate_recommendations(evaluation_report)
        }
        
        return assessment
        
    except Exception as e:
        return {'error': str(e)}


def generate_recommendations(evaluation_report):
    """
    Generate recommendations based on evaluation results
    
    Args:
        evaluation_report: Complete evaluation report
    
    Returns:
        List of recommendations
    """
    recommendations = []
    
    try:
        psnr_value = evaluation_report.get('psnr', 0)
        ssim_value = evaluation_report.get('ssim', {}).get('ssim', 0)
        
        if psnr_value < 30:
            recommendations.append("PSNR is low. Consider reducing payload size or adjusting embedding parameters.")
        
        if ssim_value < 0.9:
            recommendations.append("SSIM is low. Consider using edge-based embedding to preserve structural similarity.")
        
        reversibility = evaluation_report.get('reversibility', {})
        if not reversibility.get('perfect_reversibility', True):
            recommendations.append("Reversibility is not perfect. Check trace matrix integrity.")
        
        entropy_diff = abs(
            evaluation_report.get('entropy', {}).get('cover', {}).get('entropy', 0) -
            evaluation_report.get('entropy', {}).get('stego', {}).get('entropy', 0)
        )
        if entropy_diff > 0.5:
            recommendations.append("Entropy change is significant. Consider better pixel selection strategy.")
        
        hist_corr = evaluation_report.get('histogram_similarity', {}).get('correlation', 0)
        if hist_corr < 0.95:
            recommendations.append("Histogram correlation is low. Embedding may be detectable.")
        
    except Exception:
        recommendations.append("Unable to generate specific recommendations due to evaluation errors.")
    
    return recommendations


def save_evaluation_report(evaluation_report, output_path):
    """
    Save evaluation report to JSON file
    
    Args:
        evaluation_report: Complete evaluation report
        output_path: Path where to save the report
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert the report
    serializable_report = convert_numpy_types(evaluation_report)
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_report, f, indent=2, ensure_ascii=False)


def create_evaluation_plots(cover_image, stego_image, output_dir):
    """
    Create visualization plots for evaluation
    
    Args:
        cover_image: Original cover image
        stego_image: Stego image
        output_dir: Directory to save plots
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to grayscale for analysis
    if len(cover_image.shape) == 3:
        cover_gray = cv2.cvtColor(cover_image, cv2.COLOR_BGR2GRAY)
        stego_gray = cv2.cvtColor(stego_image, cv2.COLOR_BGR2GRAY)
    else:
        cover_gray = cover_image
        stego_gray = stego_image
    
    # Histogram comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(cover_gray.flatten(), bins=256, alpha=0.7, label='Cover', color='blue')
    plt.hist(stego_gray.flatten(), bins=256, alpha=0.7, label='Stego', color='red')
    plt.title('Histogram Comparison')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Difference image
    plt.subplot(1, 3, 2)
    diff_image = np.abs(cover_gray.astype(np.int32) - stego_gray.astype(np.int32))
    plt.imshow(diff_image, cmap='hot', vmin=0, vmax=10)
    plt.title('Difference Image')
    plt.colorbar()
    
    # SSIM map
    plt.subplot(1, 3, 3)
    ssim_map = ssim(cover_gray, stego_gray, full=True, data_range=255)[1]
    plt.imshow(ssim_map, cmap='viridis', vmin=0, vmax=1)
    plt.title('SSIM Map')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()


def batch_evaluate(cover_images, stego_images, output_dir, trace_matrices=None):
    """
    Batch evaluation of multiple image pairs
    
    Args:
        cover_images: List of cover images or paths
        stego_images: List of stego images or paths
        output_dir: Output directory for reports
        trace_matrices: List of trace matrices (optional)
    
    Returns:
        Batch evaluation results
    """
    import os
    from .utils import load_image
    
    os.makedirs(output_dir, exist_ok=True)
    batch_results = []
    
    for i, (cover, stego) in enumerate(zip(cover_images, stego_images)):
        try:
            # Load images if they are paths
            if isinstance(cover, str):
                cover_img = load_image(cover)
                image_name = os.path.splitext(os.path.basename(cover))[0]
            else:
                cover_img = cover
                image_name = f"image_{i}"
            
            if isinstance(stego, str):
                stego_img = load_image(stego)
            else:
                stego_img = stego
            
            # Get trace matrix if available
            trace_matrix = None
            if trace_matrices is not None and i < len(trace_matrices):
                trace_matrix = trace_matrices[i]
            
            # Evaluate
            evaluation_report = evaluate_metrics(cover_img, stego_img, trace_matrix=trace_matrix)
            
            # Save individual report
            report_path = os.path.join(output_dir, f"{image_name}_evaluation.json")
            save_evaluation_report(evaluation_report, report_path)
            
            # Create plots
            plots_dir = os.path.join(output_dir, f"{image_name}_plots")
            create_evaluation_plots(cover_img, stego_img, plots_dir)
            
            result = {
                'image_name': image_name,
                'report_path': report_path,
                'plots_dir': plots_dir,
                'evaluation_summary': evaluation_report.get('quality_assessment', {}),
                'success': True
            }
            
        except Exception as e:
            result = {
                'image_name': image_name if 'image_name' in locals() else f"image_{i}",
                'error': str(e),
                'success': False
            }
        
        batch_results.append(result)
    
    # Save batch summary
    summary_path = os.path.join(output_dir, 'batch_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    return batch_results