"""
Comprehensive MRI Steganography Evaluation Module
=================================================

This module provides comprehensive evaluation metrics for MRI steganography including:
- Image Quality Metrics: PSNR, SSIM, MSE
- Security Metrics: UACI, NPCR  
- Performance Metrics: Embedding capacity, extraction accuracy, runtime
- Clinical Validation: Radiologist assessment framework

Author: MRI Steganography Research Team
Date: October 2025
"""

import numpy as np
import cv2
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class ComprehensiveMRIEvaluator:
    """Comprehensive evaluation framework for MRI steganography"""
    
    def __init__(self, output_dir: str = "results/comprehensive_evaluation"):
        """
        Initialize comprehensive evaluator
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize result storage
        self.results = {
            'image_quality_metrics': [],
            'security_metrics': [], 
            'performance_metrics': [],
            'clinical_validation': [],
            'runtime_analysis': [],
            'summary_statistics': {}
        }
    
    def calculate_image_quality_metrics(self, original: np.ndarray, stego: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive image quality metrics
        
        Args:
            original: Original MRI image
            stego: Steganographic image
            
        Returns:
            Dictionary of image quality metrics
        """
        # Ensure images are in the same format
        if len(original.shape) == 3 and len(stego.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            stego_gray = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)
        elif len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            stego_gray = stego
        elif len(stego.shape) == 3:
            original_gray = original
            stego_gray = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original
            stego_gray = stego
        
        # Convert to float for calculations
        original_float = original_gray.astype(np.float64)
        stego_float = stego_gray.astype(np.float64)
        
        metrics = {}
        
        # 1. Mean Squared Error (MSE)
        mse = np.mean((original_float - stego_float) ** 2)
        metrics['mse'] = float(mse)
        
        # 2. Peak Signal-to-Noise Ratio (PSNR)
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        metrics['psnr'] = float(psnr)
        
        # 3. Structural Similarity Index (SSIM)
        ssim = self._calculate_ssim(original_gray, stego_gray)
        metrics['ssim'] = float(ssim)
        
        # 4. Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        metrics['rmse'] = float(rmse)
        
        # 5. Mean Absolute Error (MAE)
        mae = np.mean(np.abs(original_float - stego_float))
        metrics['mae'] = float(mae)
        
        # 6. Normalized Cross-Correlation (NCC)
        ncc = self._calculate_ncc(original_float, stego_float)
        metrics['ncc'] = float(ncc)
        
        # 7. Universal Image Quality Index (UIQI)
        uiqi = self._calculate_uiqi(original_float, stego_float)
        metrics['uiqi'] = float(uiqi)
        
        return metrics
    
    def calculate_security_metrics(self, original: np.ndarray, stego: np.ndarray) -> Dict[str, float]:
        """
        Calculate security analysis metrics
        
        Args:
            original: Original MRI image
            stego: Steganographic image
            
        Returns:
            Dictionary of security metrics
        """
        # Ensure grayscale
        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        if len(stego.shape) == 3:
            stego = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)
        
        metrics = {}
        
        # 1. Number of Pixels Change Rate (NPCR)
        npcr = self._calculate_npcr(original, stego)
        metrics['npcr'] = float(npcr)
        
        # 2. Unified Average Changing Intensity (UACI)
        uaci = self._calculate_uaci(original, stego)
        metrics['uaci'] = float(uaci)
        
        # 3. Correlation Analysis
        correlation = self._calculate_correlation_analysis(original, stego)
        metrics.update(correlation)
        
        # 4. Histogram Analysis
        hist_analysis = self._calculate_histogram_analysis(original, stego)
        metrics.update(hist_analysis)
        
        # 5. Entropy Analysis
        entropy_analysis = self._calculate_entropy_analysis(original, stego)
        metrics.update(entropy_analysis)
        
        return metrics
    
    def calculate_performance_metrics(self, original: np.ndarray, payload_size: int, 
                                    extracted_payload: bytes, original_payload: bytes,
                                    embed_time: float, extract_time: float) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Args:
            original: Original MRI image
            payload_size: Size of embedded payload in bits
            extracted_payload: Extracted payload
            original_payload: Original payload
            embed_time: Embedding time in seconds
            extract_time: Extraction time in seconds
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        # 1. Embedding Capacity
        image_size = original.shape[0] * original.shape[1]
        if len(original.shape) == 3:
            image_size *= original.shape[2]
        
        capacity_ratio = payload_size / (image_size * 8)  # bits per bit
        capacity_bpp = payload_size / (original.shape[0] * original.shape[1])  # bits per pixel
        
        metrics['embedding_capacity_ratio'] = float(capacity_ratio)
        metrics['embedding_capacity_bpp'] = float(capacity_bpp)
        metrics['payload_size_bits'] = int(payload_size)
        metrics['payload_size_bytes'] = int(payload_size // 8)
        
        # 2. Extraction Accuracy
        if len(extracted_payload) > 0 and len(original_payload) > 0:
            # Bit-level accuracy
            min_len = min(len(extracted_payload), len(original_payload))
            bit_accuracy = sum(a == b for a, b in zip(extracted_payload[:min_len], 
                                                     original_payload[:min_len])) / min_len
            
            # Byte-level accuracy  
            byte_accuracy = 1.0 if extracted_payload == original_payload else 0.0
            
            metrics['extraction_accuracy_bit'] = float(bit_accuracy)
            metrics['extraction_accuracy_byte'] = float(byte_accuracy)
        else:
            metrics['extraction_accuracy_bit'] = 0.0
            metrics['extraction_accuracy_byte'] = 0.0
        
        # 3. Runtime Performance
        metrics['embedding_time_seconds'] = float(embed_time)
        metrics['extraction_time_seconds'] = float(extract_time)
        metrics['total_processing_time'] = float(embed_time + extract_time)
        
        # 4. Throughput Metrics
        if embed_time > 0:
            embed_throughput = payload_size / embed_time  # bits per second
            metrics['embedding_throughput_bps'] = float(embed_throughput)
        
        if extract_time > 0:
            extract_throughput = payload_size / extract_time  # bits per second
            metrics['extraction_throughput_bps'] = float(extract_throughput)
        
        return metrics
    
    def setup_radiologist_validation_framework(self, validation_dir: str) -> Dict[str, Any]:
        """
        Setup framework for radiologist validation
        
        Args:
            validation_dir: Directory containing validation MRI images
            
        Returns:
            Validation framework configuration
        """
        framework = {
            'validation_protocol': {
                'image_count': 50,
                'evaluation_criteria': [
                    'overall_image_quality',
                    'diagnostic_confidence',
                    'anatomical_structure_visibility',
                    'lesion_detectability',
                    'noise_perception',
                    'artifact_presence'
                ],
                'rating_scale': {
                    'range': '1-5',
                    'description': {
                        '1': 'Poor - Non-diagnostic quality',
                        '2': 'Fair - Limited diagnostic value', 
                        '3': 'Good - Adequate for diagnosis',
                        '4': 'Very Good - High diagnostic confidence',
                        '5': 'Excellent - Optimal diagnostic quality'
                    }
                }
            },
            'validation_workflow': {
                'randomization': True,
                'blinding': True,
                'comparison_method': 'paired',
                'presentation_order': 'randomized'
            },
            'data_collection': {
                'original_scores': [],
                'stego_scores': [],
                'radiologist_feedback': [],
                'processing_time': []
            }
        }
        
        # Create validation templates
        self._create_validation_templates(validation_dir, framework)
        
        return framework
    
    def analyze_radiologist_validation(self, validation_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze radiologist validation results
        
        Args:
            validation_results: Results from radiologist evaluation
            
        Returns:
            Statistical analysis of validation results
        """
        analysis = {}
        
        original_scores = validation_results['data_collection']['original_scores']
        stego_scores = validation_results['data_collection']['stego_scores']
        
        if len(original_scores) > 0 and len(stego_scores) > 0:
            # Descriptive statistics
            analysis['original_mean'] = float(np.mean(original_scores))
            analysis['original_std'] = float(np.std(original_scores))
            analysis['stego_mean'] = float(np.mean(stego_scores))
            analysis['stego_std'] = float(np.std(stego_scores))
            
            # Diagnostic accuracy preservation
            # Assume scores >= 3 are diagnostically acceptable
            original_diagnostic = np.array(original_scores) >= 3
            stego_diagnostic = np.array(stego_scores) >= 3
            
            original_accuracy = np.mean(original_diagnostic) * 100
            stego_accuracy = np.mean(stego_diagnostic) * 100
            accuracy_preservation = (stego_accuracy / original_accuracy) * 100 if original_accuracy > 0 else 0
            
            analysis['original_diagnostic_accuracy'] = float(original_accuracy)
            analysis['stego_diagnostic_accuracy'] = float(stego_accuracy)
            analysis['accuracy_preservation_percent'] = float(accuracy_preservation)
            
            # Statistical significance testing
            if len(original_scores) == len(stego_scores):
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(original_scores, stego_scores)
                analysis['paired_ttest_statistic'] = float(t_stat)
                analysis['paired_ttest_pvalue'] = float(p_value)
                analysis['statistically_significant'] = p_value < 0.05
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(original_scores) + np.var(stego_scores)) / 2)
                cohens_d = (np.mean(original_scores) - np.mean(stego_scores)) / pooled_std if pooled_std > 0 else 0
                analysis['effect_size_cohens_d'] = float(cohens_d)
            
            # Agreement analysis
            agreement_matrix = confusion_matrix(original_diagnostic, stego_diagnostic)
            analysis['agreement_matrix'] = agreement_matrix.tolist()
            
            # Overall agreement percentage
            agreement_percent = np.mean(original_diagnostic == stego_diagnostic) * 100
            analysis['overall_agreement_percent'] = float(agreement_percent)
        
        return analysis
    
    def generate_comprehensive_report(self, image_path: str, evaluation_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            image_path: Path to evaluated MRI image
            evaluation_results: Complete evaluation results
            
        Returns:
            Path to generated report
        """
        report_data = {
            'evaluation_metadata': {
                'image_path': image_path,
                'evaluation_timestamp': datetime.now().isoformat(),
                'image_dimensions': evaluation_results.get('image_dimensions', 'Unknown'),
                'image_format': os.path.splitext(image_path)[1]
            },
            'results': evaluation_results
        }
        
        # Save detailed JSON report
        report_path = os.path.join(self.output_dir, f'comprehensive_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate summary report
        summary_path = self._generate_summary_report(report_data)
        
        # Generate visualizations
        viz_path = self._generate_visualization_report(report_data)
        
        print(f"📄 Comprehensive report saved: {report_path}")
        print(f"📊 Summary report saved: {summary_path}")
        print(f"📈 Visualization report saved: {viz_path}")
        
        return report_path
    
    # Helper methods
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM between two images"""
        try:
            from skimage.metrics import structural_similarity
            return structural_similarity(img1, img2, data_range=255)
        except ImportError:
            # Fallback implementation
            return self._ssim_fallback(img1, img2)
    
    def _ssim_fallback(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Fallback SSIM implementation"""
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        k1, k2 = 0.01, 0.03
        L = 255
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        return float(ssim)
    
    def _calculate_ncc(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Normalized Cross-Correlation"""
        img1_norm = img1 - np.mean(img1)
        img2_norm = img2 - np.mean(img2)
        
        numerator = np.sum(img1_norm * img2_norm)
        denominator = np.sqrt(np.sum(img1_norm**2) * np.sum(img2_norm**2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_uiqi(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Universal Image Quality Index"""
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        numerator = 4 * sigma12 * mu1 * mu2
        denominator = (sigma1_sq + sigma2_sq) * (mu1**2 + mu2**2)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_npcr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Number of Pixels Change Rate"""
        diff = img1 != img2
        npcr = np.sum(diff) / img1.size * 100
        return npcr
    
    def _calculate_uaci(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Unified Average Changing Intensity"""
        diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64))
        uaci = np.mean(diff) / 255 * 100
        return uaci
    
    def _calculate_correlation_analysis(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
        """Calculate correlation analysis metrics"""
        correlation = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
        
        return {
            'pixel_correlation': float(correlation),
            'correlation_strength': 'strong' if abs(correlation) > 0.9 else 'moderate' if abs(correlation) > 0.7 else 'weak'
        }
    
    def _calculate_histogram_analysis(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
        """Calculate histogram-based analysis"""
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256]).flatten()
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256]).flatten()
        
        # Chi-square distance
        chi_square = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        
        # Bhattacharyya distance
        bhattacharyya = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        
        return {
            'histogram_chi_square': float(chi_square),
            'histogram_bhattacharyya': float(bhattacharyya)
        }
    
    def _calculate_entropy_analysis(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
        """Calculate entropy analysis"""
        def calculate_entropy(image):
            hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
            hist = hist / np.sum(hist)  # Normalize
            hist = hist[hist > 0]  # Remove zeros
            return -np.sum(hist * np.log2(hist))
        
        entropy1 = calculate_entropy(img1)
        entropy2 = calculate_entropy(img2)
        
        return {
            'original_entropy': float(entropy1),
            'stego_entropy': float(entropy2),
            'entropy_difference': float(abs(entropy1 - entropy2))
        }
    
    def _create_validation_templates(self, validation_dir: str, framework: Dict[str, Any]):
        """Create templates for radiologist validation"""
        # Create validation form template
        form_template = {
            'instructions': 'Please evaluate each MRI image pair and rate the diagnostic quality.',
            'evaluation_criteria': framework['validation_protocol']['evaluation_criteria'],
            'rating_scale': framework['validation_protocol']['rating_scale'],
            'data_entry': []
        }
        
        template_path = os.path.join(validation_dir, 'validation_form_template.json')
        with open(template_path, 'w') as f:
            json.dump(form_template, f, indent=2)
    
    def _generate_summary_report(self, report_data: Dict[str, Any]) -> str:
        """Generate human-readable summary report"""
        summary_path = os.path.join(self.output_dir, 'evaluation_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("MRI STEGANOGRAPHY COMPREHENSIVE EVALUATION SUMMARY\n")
            f.write("=" * 55 + "\n\n")
            
            # Write summary sections
            if 'image_quality_metrics' in report_data['results']:
                f.write("IMAGE QUALITY METRICS:\n")
                f.write("-" * 25 + "\n")
                metrics = report_data['results']['image_quality_metrics']
                f.write(f"PSNR: {metrics.get('psnr', 'N/A'):.2f} dB\n")
                f.write(f"SSIM: {metrics.get('ssim', 'N/A'):.4f}\n")
                f.write(f"MSE: {metrics.get('mse', 'N/A'):.2f}\n")
                f.write(f"RMSE: {metrics.get('rmse', 'N/A'):.2f}\n")
                f.write("\n")
            
            if 'security_metrics' in report_data['results']:
                f.write("SECURITY METRICS:\n")
                f.write("-" * 17 + "\n")
                metrics = report_data['results']['security_metrics']
                f.write(f"NPCR: {metrics.get('npcr', 'N/A'):.2f}%\n")
                f.write(f"UACI: {metrics.get('uaci', 'N/A'):.2f}%\n")
                f.write("\n")
            
            if 'performance_metrics' in report_data['results']:
                f.write("PERFORMANCE METRICS:\n")
                f.write("-" * 20 + "\n")
                metrics = report_data['results']['performance_metrics']
                f.write(f"Embedding Capacity: {metrics.get('embedding_capacity_bpp', 'N/A'):.4f} bpp\n")
                f.write(f"Extraction Accuracy: {metrics.get('extraction_accuracy_byte', 'N/A'):.2%}\n")
                f.write(f"Runtime: {metrics.get('total_processing_time', 'N/A'):.3f} seconds\n")
                f.write("\n")
        
        return summary_path
    
    def _generate_visualization_report(self, report_data: Dict[str, Any]) -> str:
        """Generate visualization report"""
        viz_path = os.path.join(self.output_dir, 'evaluation_visualizations.png')
        
        # Create visualization plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('MRI Steganography Evaluation Results', fontsize=16)
        
        # Placeholder visualizations (implement based on available data)
        axes[0, 0].text(0.5, 0.5, 'Image Quality\nMetrics', ha='center', va='center', fontsize=12)
        axes[0, 1].text(0.5, 0.5, 'Security\nAnalysis', ha='center', va='center', fontsize=12)
        axes[1, 0].text(0.5, 0.5, 'Performance\nMetrics', ha='center', va='center', fontsize=12)
        axes[1, 1].text(0.5, 0.5, 'Clinical\nValidation', ha='center', va='center', fontsize=12)
        
        for ax in axes.flat:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path


class MRIBatchEvaluator:
    """Batch evaluation for multiple MRI images"""
    
    def __init__(self, dataset_dir: str, output_dir: str = "results/batch_evaluation"):
        """
        Initialize batch evaluator
        
        Args:
            dataset_dir: Directory containing MRI dataset
            output_dir: Directory to save batch evaluation results
        """
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.evaluator = ComprehensiveMRIEvaluator(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
    
    def run_batch_evaluation(self, max_images: int = 50) -> Dict[str, Any]:
        """
        Run batch evaluation on MRI dataset
        
        Args:
            max_images: Maximum number of images to evaluate
            
        Returns:
            Batch evaluation results
        """
        print(f"🔍 Starting batch evaluation on {self.dataset_dir}")
        print(f"📁 Results will be saved to: {self.output_dir}")
        
        # Find MRI images
        image_files = self._find_mri_images(max_images)
        
        if not image_files:
            raise ValueError(f"No MRI images found in {self.dataset_dir}")
        
        print(f"📊 Found {len(image_files)} MRI images for evaluation")
        
        batch_results = {
            'evaluation_metadata': {
                'dataset_dir': self.dataset_dir,
                'total_images': len(image_files),
                'evaluation_timestamp': datetime.now().isoformat()
            },
            'individual_results': [],
            'aggregate_statistics': {},
            'validation_framework': {}
        }
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            print(f"\n🔬 Evaluating image {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            try:
                result = self._evaluate_single_image(image_path)
                batch_results['individual_results'].append(result)
                print(f"✅ Completed evaluation for {os.path.basename(image_path)}")
                
            except Exception as e:
                print(f"❌ Error evaluating {os.path.basename(image_path)}: {str(e)}")
                continue
        
        # Calculate aggregate statistics
        if batch_results['individual_results']:
            batch_results['aggregate_statistics'] = self._calculate_aggregate_statistics(
                batch_results['individual_results']
            )
        
        # Setup validation framework for radiologist evaluation
        batch_results['validation_framework'] = self.evaluator.setup_radiologist_validation_framework(
            self.dataset_dir
        )
        
        # Save batch results
        results_path = os.path.join(self.output_dir, 'batch_evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        print(f"\n📄 Batch evaluation results saved: {results_path}")
        
        return batch_results
    
    def _find_mri_images(self, max_images: int) -> List[str]:
        """Find MRI images in dataset directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
        image_files = []
        
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
                    
                    if len(image_files) >= max_images:
                        break
            
            if len(image_files) >= max_images:
                break
        
        return image_files[:max_images]
    
    def _evaluate_single_image(self, image_path: str) -> Dict[str, Any]:
        """Evaluate a single MRI image"""
        # Load original image
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Ensure 512x512 size
        if original.shape[:2] != (512, 512):
            original = cv2.resize(original, (512, 512))
        
        # Simulate steganography process (placeholder - integrate with actual system)
        start_time = time.time()
        
        # TODO: Integrate with actual steganography embedding
        # For now, simulate by adding minimal noise
        stego = original.copy()
        noise = np.random.randint(-1, 2, original.shape, dtype=np.int8)
        stego = np.clip(original.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        embed_time = time.time() - start_time
        
        # Simulate extraction
        start_time = time.time()
        # TODO: Integrate with actual extraction
        extracted_payload = b"Sample medical text for testing"
        original_payload = b"Sample medical text for testing"
        extract_time = time.time() - start_time
        
        # Calculate metrics
        result = {
            'image_path': image_path,
            'image_dimensions': original.shape,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Image quality metrics
        result['image_quality_metrics'] = self.evaluator.calculate_image_quality_metrics(
            original, stego
        )
        
        # Security metrics
        result['security_metrics'] = self.evaluator.calculate_security_metrics(
            original, stego
        )
        
        # Performance metrics
        payload_size = len(original_payload) * 8  # bits
        result['performance_metrics'] = self.evaluator.calculate_performance_metrics(
            original, payload_size, extracted_payload, original_payload,
            embed_time, extract_time
        )
        
        return result
    
    def _calculate_aggregate_statistics(self, individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate statistics across all evaluated images"""
        if not individual_results:
            return {}
        
        # Collect metrics from all results
        quality_metrics = []
        security_metrics = []
        performance_metrics = []
        
        for result in individual_results:
            if 'image_quality_metrics' in result:
                quality_metrics.append(result['image_quality_metrics'])
            if 'security_metrics' in result:
                security_metrics.append(result['security_metrics'])
            if 'performance_metrics' in result:
                performance_metrics.append(result['performance_metrics'])
        
        aggregate = {}
        
        # Calculate statistics for each metric category
        if quality_metrics:
            aggregate['image_quality_statistics'] = self._calculate_metric_statistics(quality_metrics)
        
        if security_metrics:
            aggregate['security_statistics'] = self._calculate_metric_statistics(security_metrics)
        
        if performance_metrics:
            aggregate['performance_statistics'] = self._calculate_metric_statistics(performance_metrics)
        
        return aggregate
    
    def _calculate_metric_statistics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for a list of metric dictionaries"""
        if not metrics_list:
            return {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        statistics = {}
        
        for key in all_keys:
            values = [metrics.get(key, np.nan) for metrics in metrics_list]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                statistics[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'count': len(values)
                }
        
        return statistics