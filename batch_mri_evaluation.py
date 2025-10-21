#!/usr/bin/env python3
"""
Batch MRI Evaluation Script
===========================

Comprehensive evaluation script for MRI steganography on multiple images.
Evaluates PSNR, SSIM, MSE, UACI, NPCR, embedding capacity, extraction accuracy,
and runtime performance.

Usage:
    python batch_mri_evaluation.py --dataset_dir data/mri_dataset/test_images/
    python batch_mri_evaluation.py --validation_dir data/mri_dataset/validation_set/ --radiologist_mode

Author: MRI Steganography Research Team
Date: October 2025
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.comprehensive_evaluation import ComprehensiveMRIEvaluator, MRIBatchEvaluator
from src.embedding import embed_payload, extract_payload
from src.utils import load_image, save_image, save_trace_matrix, load_trace_matrix
from src.mri_preprocessing import MRIPreprocessor
from src.advanced_lbp import AdvancedLBP
from src.roi_adaptive_embedding import ROIAdaptiveEmbedding
from src.clinical_evaluation import ClinicalEvaluator


class MRISteganoEvaluationPipeline:
    """Complete evaluation pipeline for MRI steganography"""
    
    def __init__(self, output_dir: str = "results/mri_evaluation"):
        """
        Initialize evaluation pipeline
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        self.evaluator = ComprehensiveMRIEvaluator(output_dir)
        self.preprocessor = MRIPreprocessor()
        self.lbp_analyzer = AdvancedLBP()
        self.roi_embedder = ROIAdaptiveEmbedding()
        self.clinical_validator = ClinicalEvaluator()
        
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate_single_mri(self, image_path: str, payload_text: str, 
                           enable_mri_features: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation of single MRI image
        
        Args:
            image_path: Path to MRI image
            payload_text: Text payload to embed
            enable_mri_features: Whether to use MRI-specific features
            
        Returns:
            Complete evaluation results
        """
        print(f"🔬 Evaluating MRI image: {os.path.basename(image_path)}")
        
        # Load and validate image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Ensure 512x512
        if original_image.shape[:2] != (512, 512):
            original_image = cv2.resize(original_image, (512, 512))
            print(f"   Resized image to 512x512")
        
        # Convert payload to bytes
        payload_bytes = payload_text.encode('utf-8')
        payload_size = len(payload_bytes) * 8  # bits
        
        print(f"   Original image shape: {original_image.shape}")
        print(f"   Payload size: {len(payload_bytes)} bytes ({payload_size} bits)")
        
        results = {
            'metadata': {
                'image_path': image_path,
                'original_dimensions': original_image.shape,
                'payload_size_bytes': len(payload_bytes),
                'payload_size_bits': payload_size,
                'mri_features_enabled': enable_mri_features,
                'evaluation_timestamp': datetime.now().isoformat()
            }
        }
        
        try:
            # Step 1: MRI Preprocessing (if enabled)
            if enable_mri_features:
                print("   🧠 Applying MRI-specific preprocessing...")
                start_time = time.time()
                preprocessed_image, preprocessing_info = self.preprocessor.preprocess_mri(
                    original_image, 
                    enable_denoising=True,
                    enable_bias_correction=True,
                    enable_normalization=True
                )
                preprocessing_time = time.time() - start_time
                print(f"   Preprocessing completed in {preprocessing_time:.3f}s")
                
                results['preprocessing'] = {
                    'applied': True,
                    'processing_time': preprocessing_time,
                    'info': preprocessing_info
                }
            else:
                preprocessed_image = original_image.copy()
                results['preprocessing'] = {'applied': False}
            
            # Step 2: Advanced Texture Analysis
            print("   🔍 Performing advanced texture analysis...")
            start_time = time.time()
            
            if len(preprocessed_image.shape) == 3:
                gray_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = preprocessed_image
            
            texture_features = self.lbp_analyzer.extract_comprehensive_features(gray_image)
            texture_time = time.time() - start_time
            print(f"   Texture analysis completed in {texture_time:.3f}s")
            
            # Step 3: ROI-Adaptive Embedding (if enabled)
            if enable_mri_features:
                print("   🎯 Performing ROI-adaptive embedding...")
                start_time = time.time()
                
                # Generate edge map (simplified)
                edge_map = cv2.Canny(gray_image, 50, 150).astype(np.float32) / 255.0
                texture_map = texture_features['texture_strength_map']
                
                selected_pixels, roi_info = self.roi_embedder.adaptive_pixel_selection(
                    gray_image, edge_map, texture_map, payload_size
                )
                
                roi_time = time.time() - start_time
                print(f"   ROI analysis completed in {roi_time:.3f}s")
                print(f"   Selected {roi_info['selection_info']['n_selected']} pixels for embedding")
                
                results['roi_analysis'] = {
                    'applied': True,
                    'processing_time': roi_time,
                    'selected_pixels': int(roi_info['selection_info']['n_selected']),
                    'safety_ratio': float(roi_info['selection_info']['safety_ratio']),
                    'roi_info': roi_info
                }
            else:
                results['roi_analysis'] = {'applied': False}
            
            # Step 4: Steganographic Embedding
            print("   🔒 Performing steganographic embedding...")
            start_time = time.time()
            
            # Embedding configuration
            embedding_config = {
                'method': 'hybrid_lsb',
                'edge_threshold': 0.2,
                'texture_threshold': 0.5,
                'mri_mode': enable_mri_features
            }
            
            # Perform embedding (using existing embedding function)
            try:
                stego_image, trace_matrix, embedding_stats = embed_payload(
                    preprocessed_image, payload_bytes, embedding_config
                )
                embed_time = time.time() - start_time
                print(f"   Embedding completed in {embed_time:.3f}s")
                
                # Step 5: Payload Extraction
                print("   🔓 Extracting embedded payload...")
                start_time = time.time()
                
                extracted_payload, payload_metadata, extraction_info = extract_payload(
                    stego_image, trace_matrix=trace_matrix, config=embedding_config
                )
                extract_time = time.time() - start_time
                print(f"   Extraction completed in {extract_time:.3f}s")
                
                # Convert extracted payload for comparison
                if isinstance(extracted_payload, bytes):
                    extracted_text = extracted_payload.decode('utf-8', errors='ignore')
                else:
                    extracted_text = str(extracted_payload)
                
                # Step 6: Comprehensive Metric Calculation
                print("   📊 Calculating comprehensive metrics...")
                
                # Image Quality Metrics
                quality_metrics = self.evaluator.calculate_image_quality_metrics(
                    preprocessed_image, stego_image
                )
                
                # Security Metrics
                security_metrics = self.evaluator.calculate_security_metrics(
                    preprocessed_image, stego_image
                )
                
                # Performance Metrics
                performance_metrics = self.evaluator.calculate_performance_metrics(
                    preprocessed_image, payload_size, extracted_payload, payload_bytes,
                    embed_time, extract_time
                )
                
                # Step 7: Clinical Evaluation (if enabled)
                if enable_mri_features:
                    print("   🏥 Performing clinical evaluation...")
                    start_time = time.time()
                    
                    clinical_report = self.clinical_validator.comprehensive_clinical_evaluation(
                        preprocessed_image, stego_image, 
                        enable_visual_grading=True,
                        enable_diagnostic_assessment=True
                    )
                    clinical_time = time.time() - start_time
                    print(f"   Clinical evaluation completed in {clinical_time:.3f}s")
                    
                    results['clinical_evaluation'] = {
                        'applied': True,
                        'processing_time': clinical_time,
                        'report': clinical_report
                    }
                else:
                    results['clinical_evaluation'] = {'applied': False}
                
                # Compile all results
                results.update({
                    'embedding_stats': embedding_stats,
                    'extraction_info': extraction_info,
                    'image_quality_metrics': quality_metrics,
                    'security_metrics': security_metrics,
                    'performance_metrics': performance_metrics,
                    'payload_verification': {
                        'original_payload': payload_text,
                        'extracted_payload': extracted_text,
                        'exact_match': payload_text == extracted_text,
                        'similarity_ratio': self._calculate_text_similarity(payload_text, extracted_text)
                    }
                })
                
                print(f"✅ Evaluation completed successfully")
                print(f"   PSNR: {quality_metrics['psnr']:.2f} dB")
                print(f"   SSIM: {quality_metrics['ssim']:.4f}")
                print(f"   UACI: {security_metrics['uaci']:.2f}%")
                print(f"   NPCR: {security_metrics['npcr']:.2f}%")
                print(f"   Extraction Accuracy: {performance_metrics['extraction_accuracy_byte']:.2%}")
                
                return results
                
            except Exception as e:
                print(f"❌ Error during steganographic process: {str(e)}")
                results['error'] = str(e)
                return results
                
        except Exception as e:
            print(f"❌ Error during evaluation: {str(e)}")
            results['error'] = str(e)
            return results
    
    def run_batch_evaluation(self, dataset_dir: str, max_images: int = 50, 
                           enable_mri_features: bool = True) -> Dict[str, Any]:
        """
        Run batch evaluation on multiple MRI images
        
        Args:
            dataset_dir: Directory containing MRI images
            max_images: Maximum number of images to evaluate
            enable_mri_features: Whether to use MRI-specific features
            
        Returns:
            Batch evaluation results
        """
        print(f"🔍 Starting batch MRI evaluation")
        print(f"📁 Dataset directory: {dataset_dir}")
        print(f"📊 Maximum images: {max_images}")
        print(f"🧠 MRI features enabled: {enable_mri_features}")
        
        # Find MRI images
        image_files = self._find_mri_images(dataset_dir, max_images)
        
        if not image_files:
            raise ValueError(f"No MRI images found in {dataset_dir}")
        
        print(f"📷 Found {len(image_files)} MRI images")
        
        # Sample payload for testing
        sample_payload = """
        Patient ID: TEST_001
        Scan Date: 2025-10-21
        Sequence: T1-weighted MPRAGE
        Notes: High-resolution anatomical scan for research purposes.
        Quality: Diagnostic grade
        """.strip()
        
        batch_results = {
            'evaluation_metadata': {
                'dataset_dir': dataset_dir,
                'total_images': len(image_files),
                'mri_features_enabled': enable_mri_features,
                'evaluation_timestamp': datetime.now().isoformat(),
                'sample_payload': sample_payload
            },
            'individual_results': [],
            'aggregate_statistics': {},
            'summary_report': {}
        }
        
        # Evaluate each image
        successful_evaluations = 0
        total_start_time = time.time()
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n{'='*60}")
            print(f"🔬 Evaluating image {i}/{len(image_files)}")
            
            try:
                result = self.evaluate_single_mri(image_path, sample_payload, enable_mri_features)
                
                if 'error' not in result:
                    batch_results['individual_results'].append(result)
                    successful_evaluations += 1
                    print(f"✅ Successfully evaluated {os.path.basename(image_path)}")
                else:
                    print(f"❌ Failed to evaluate {os.path.basename(image_path)}: {result['error']}")
                
            except Exception as e:
                print(f"❌ Error evaluating {os.path.basename(image_path)}: {str(e)}")
                continue
        
        total_evaluation_time = time.time() - total_start_time
        
        print(f"\n{'='*60}")
        print(f"📊 Batch evaluation completed")
        print(f"✅ Successful evaluations: {successful_evaluations}/{len(image_files)}")
        print(f"⏱️  Total evaluation time: {total_evaluation_time:.2f} seconds")
        
        # Calculate aggregate statistics
        if batch_results['individual_results']:
            print("🔢 Calculating aggregate statistics...")
            batch_results['aggregate_statistics'] = self._calculate_comprehensive_statistics(
                batch_results['individual_results']
            )
            
            # Generate summary report
            batch_results['summary_report'] = self._generate_batch_summary(
                batch_results['individual_results'], total_evaluation_time
            )
        
        # Save results
        results_path = os.path.join(self.output_dir, f'batch_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        with open(results_path, 'w') as f:
            import json
            json.dump(batch_results, f, indent=2, default=str)
        
        print(f"📄 Batch results saved: {results_path}")
        
        # Print summary
        self._print_batch_summary(batch_results['summary_report'])
        
        return batch_results
    
    def _find_mri_images(self, dataset_dir: str, max_images: int) -> List[str]:
        """Find MRI images in dataset directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
        image_files = []
        
        if not os.path.exists(dataset_dir):
            raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
        
        for root, dirs, files in os.walk(dataset_dir):
            for file in sorted(files):  # Sort for consistent ordering
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
                    
                    if len(image_files) >= max_images:
                        break
            
            if len(image_files) >= max_images:
                break
        
        return image_files[:max_images]
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two text strings"""
        if not text1 or not text2:
            return 0.0
        
        # Simple character-level similarity
        min_len = min(len(text1), len(text2))
        if min_len == 0:
            return 0.0
        
        matches = sum(c1 == c2 for c1, c2 in zip(text1[:min_len], text2[:min_len]))
        return matches / min_len
    
    def _calculate_comprehensive_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive statistics across all results"""
        if not results:
            return {}
        
        stats = {}
        
        # Collect metrics
        quality_metrics = [r['image_quality_metrics'] for r in results if 'image_quality_metrics' in r]
        security_metrics = [r['security_metrics'] for r in results if 'security_metrics' in r]
        performance_metrics = [r['performance_metrics'] for r in results if 'performance_metrics' in r]
        
        # Calculate statistics for each metric type
        if quality_metrics:
            stats['image_quality'] = self._calculate_metric_stats(quality_metrics)
        
        if security_metrics:
            stats['security'] = self._calculate_metric_stats(security_metrics)
        
        if performance_metrics:
            stats['performance'] = self._calculate_metric_stats(performance_metrics)
        
        return stats
    
    def _calculate_metric_stats(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for a list of metric dictionaries"""
        if not metrics_list:
            return {}
        
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        stats = {}
        
        for key in all_keys:
            values = []
            for metrics in metrics_list:
                if key in metrics and not np.isnan(metrics[key]):
                    values.append(metrics[key])
            
            if values:
                stats[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'count': len(values)
                }
        
        return stats
    
    def _generate_batch_summary(self, results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """Generate summary report for batch evaluation"""
        if not results:
            return {}
        
        summary = {
            'evaluation_overview': {
                'total_images': len(results),
                'total_evaluation_time': total_time,
                'average_time_per_image': total_time / len(results) if results else 0
            }
        }
        
        # Extract key metrics
        psnr_values = [r['image_quality_metrics']['psnr'] for r in results if 'image_quality_metrics' in r]
        ssim_values = [r['image_quality_metrics']['ssim'] for r in results if 'image_quality_metrics' in r]
        uaci_values = [r['security_metrics']['uaci'] for r in results if 'security_metrics' in r]
        npcr_values = [r['security_metrics']['npcr'] for r in results if 'security_metrics' in r]
        extraction_accuracy = [r['performance_metrics']['extraction_accuracy_byte'] for r in results if 'performance_metrics' in r]
        
        if psnr_values:
            summary['image_quality_summary'] = {
                'mean_psnr': np.mean(psnr_values),
                'mean_ssim': np.mean(ssim_values) if ssim_values else 0,
                'psnr_range': [np.min(psnr_values), np.max(psnr_values)],
                'ssim_range': [np.min(ssim_values), np.max(ssim_values)] if ssim_values else [0, 0]
            }
        
        if uaci_values:
            summary['security_summary'] = {
                'mean_uaci': np.mean(uaci_values),
                'mean_npcr': np.mean(npcr_values) if npcr_values else 0,
                'uaci_range': [np.min(uaci_values), np.max(uaci_values)],
                'npcr_range': [np.min(npcr_values), np.max(npcr_values)] if npcr_values else [0, 0]
            }
        
        if extraction_accuracy:
            summary['performance_summary'] = {
                'mean_extraction_accuracy': np.mean(extraction_accuracy),
                'perfect_extractions': sum(1 for acc in extraction_accuracy if acc == 1.0),
                'extraction_success_rate': sum(1 for acc in extraction_accuracy if acc > 0.9) / len(extraction_accuracy)
            }
        
        return summary
    
    def _print_batch_summary(self, summary: Dict[str, Any]):
        """Print formatted batch summary"""
        print(f"\n{'='*80}")
        print("📊 BATCH EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        if 'evaluation_overview' in summary:
            overview = summary['evaluation_overview']
            print(f"📷 Total images evaluated: {overview['total_images']}")
            print(f"⏱️  Total evaluation time: {overview['total_evaluation_time']:.2f} seconds")
            print(f"⚡ Average time per image: {overview['average_time_per_image']:.2f} seconds")
        
        if 'image_quality_summary' in summary:
            quality = summary['image_quality_summary']
            print(f"\n🖼️  IMAGE QUALITY METRICS:")
            print(f"   Mean PSNR: {quality['mean_psnr']:.2f} dB")
            print(f"   Mean SSIM: {quality['mean_ssim']:.4f}")
            print(f"   PSNR Range: {quality['psnr_range'][0]:.2f} - {quality['psnr_range'][1]:.2f} dB")
            print(f"   SSIM Range: {quality['ssim_range'][0]:.4f} - {quality['ssim_range'][1]:.4f}")
        
        if 'security_summary' in summary:
            security = summary['security_summary']
            print(f"\n🔒 SECURITY METRICS:")
            print(f"   Mean UACI: {security['mean_uaci']:.2f}%")
            print(f"   Mean NPCR: {security['mean_npcr']:.2f}%")
            print(f"   UACI Range: {security['uaci_range'][0]:.2f}% - {security['uaci_range'][1]:.2f}%")
            print(f"   NPCR Range: {security['npcr_range'][0]:.2f}% - {security['npcr_range'][1]:.2f}%")
        
        if 'performance_summary' in summary:
            performance = summary['performance_summary']
            print(f"\n⚡ PERFORMANCE METRICS:")
            print(f"   Mean Extraction Accuracy: {performance['mean_extraction_accuracy']:.2%}")
            print(f"   Perfect Extractions: {performance['perfect_extractions']}")
            print(f"   Success Rate (>90%): {performance['extraction_success_rate']:.2%}")
        
        print(f"\n{'='*80}")


def main():
    """Main function for batch MRI evaluation"""
    parser = argparse.ArgumentParser(description='Batch MRI Steganography Evaluation')
    
    parser.add_argument('--dataset_dir', type=str, 
                       default='data/mri_dataset/test_images',
                       help='Directory containing MRI images')
    
    parser.add_argument('--validation_dir', type=str,
                       default='data/mri_dataset/validation_set',
                       help='Directory for radiologist validation images')
    
    parser.add_argument('--output_dir', type=str,
                       default='results/mri_batch_evaluation',
                       help='Output directory for results')
    
    parser.add_argument('--max_images', type=int, default=50,
                       help='Maximum number of images to evaluate')
    
    parser.add_argument('--mri_features', action='store_true', default=True,
                       help='Enable MRI-specific features')
    
    parser.add_argument('--radiologist_mode', action='store_true',
                       help='Setup radiologist validation framework')
    
    args = parser.parse_args()
    
    print("🧠 MRI STEGANOGRAPHY BATCH EVALUATION")
    print("="*50)
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Maximum images: {args.max_images}")
    print(f"MRI features enabled: {args.mri_features}")
    print(f"Radiologist mode: {args.radiologist_mode}")
    
    # Initialize evaluation pipeline
    pipeline = MRISteganoEvaluationPipeline(args.output_dir)
    
    try:
        if args.radiologist_mode:
            # Setup radiologist validation framework
            print(f"\n🏥 Setting up radiologist validation framework...")
            evaluator = ComprehensiveMRIEvaluator(args.output_dir)
            framework = evaluator.setup_radiologist_validation_framework(args.validation_dir)
            
            print(f"📋 Validation framework configured for {framework['validation_protocol']['image_count']} images")
            print(f"📁 Validation templates saved to: {args.validation_dir}")
        
        # Run batch evaluation
        print(f"\n🔍 Starting batch evaluation...")
        results = pipeline.run_batch_evaluation(
            args.dataset_dir, 
            args.max_images, 
            args.mri_features
        )
        
        print(f"\n✅ Batch evaluation completed successfully!")
        print(f"📄 Results saved to: {args.output_dir}")
        
        # Additional recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Place your 512x512 JPG MRI images in: {args.dataset_dir}")
        print(f"   2. For radiologist validation, use images in: {args.validation_dir}")
        print(f"   3. Check detailed results in: {args.output_dir}")
        print(f"   4. Use --radiologist_mode for clinical validation setup")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()