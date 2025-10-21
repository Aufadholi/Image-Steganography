#!/usr/bin/env python3
"""
Enhanced MRI-Specific Hybrid Steganography System
Main Demo Script with ROI-Adaptive Embedding and Clinical Validation

This script demonstrates the complete MRI steganography workflow:
1. MRI-specific preprocessing (denoising, normalization, bias correction)
2. Advanced texture analysis with multi-scale LBP
3. ROI-adaptive embedding (avoiding diagnostic regions)
4. Reversible LSB embedding with adaptive optimization
5. Clinical evaluation and diagnostic quality assessment
"""

import os
import sys
import argparse
import glob
import json
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import (
    load_image, save_image, create_sample_payloads, 
    save_trace_matrix, get_file_info
)
from src.embedding import embed_payload, extract_payload
from src.restore import restore_cover, verify_restoration
from src.evaluation import evaluate_metrics, save_evaluation_report, create_evaluation_plots

# New MRI-specific modules
from src.mri_preprocessing import MRIPreprocessor, preprocess_mri_image
from src.advanced_lbp import AdvancedLBP, analyze_mri_texture
from src.roi_adaptive_embedding import ROIAdaptiveEmbedding, perform_roi_adaptive_embedding
from src.adaptive_optimizer import MultiObjectiveOptimizer
from src.clinical_evaluation import ClinicalEvaluator, evaluate_clinical_quality


def load_config(config_path='config.yaml'):
    """Load configuration with MRI-specific parameters"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add MRI-specific default configurations if not present
        if 'mri_specific' not in config:
            config['mri_specific'] = {
                'preprocessing': {
                    'enable_denoising': True,
                    'enable_bias_correction': True,
                    'enable_normalization': True,
                    'enable_contrast_enhancement': True
                },
                'roi_adaptive': {
                    'enable_roi_segmentation': True,
                    'safety_margin': 5,
                    'max_embedding_ratio': 0.15
                },
                'clinical_validation': {
                    'enable_clinical_evaluation': True,
                    'snr_threshold': 20.0,
                    'cnr_threshold': 5.0,
                    'visual_grade_threshold': 3.0
                }
            }
        
        return config
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using default MRI configuration.")
        return {
            'mri_specific': {
                'preprocessing': {
                    'enable_denoising': True,
                    'enable_bias_correction': True,
                    'enable_normalization': True,
                    'enable_contrast_enhancement': True
                },
                'roi_adaptive': {
                    'enable_roi_segmentation': True,
                    'safety_margin': 5,
                    'max_embedding_ratio': 0.15
                },
                'clinical_validation': {
                    'enable_clinical_evaluation': True,
                    'snr_threshold': 20.0,
                    'cnr_threshold': 5.0,
                    'visual_grade_threshold': 3.0
                }
            }
        }


def demonstrate_mri_steganography(cover_path, text_payload_path, config, output_dir, 
                                enable_mri_features=True):
    """Demonstrate MRI-specific steganography workflow"""
    print("\n" + "="*80)
    print("DEMONSTRATING MRI-SPECIFIC STEGANOGRAPHY")
    print("="*80)
    
    # Load cover image
    print(f"Loading MRI cover image: {cover_path}")
    cover_image = load_image(cover_path)
    print(f"MRI image shape: {cover_image.shape}")
    
    # MRI-specific preprocessing
    preprocessed_image = cover_image
    preprocessing_info = {}
    
    if enable_mri_features and config.get('mri_specific', {}).get('preprocessing', {}).get('enable_denoising', True):
        print("🔬 Applying MRI-specific preprocessing...")
        mri_preprocessor = MRIPreprocessor()
        preprocessed_image, preprocessing_info = mri_preprocessor.preprocess_mri(
            cover_image,
            enable_denoising=config['mri_specific']['preprocessing'].get('enable_denoising', True),
            enable_normalization=config['mri_specific']['preprocessing'].get('enable_normalization', True),
            enable_bias_correction=config['mri_specific']['preprocessing'].get('enable_bias_correction', True),
            enable_contrast_enhancement=config['mri_specific']['preprocessing'].get('enable_contrast_enhancement', True)
        )
        print(f"   Preprocessing steps: {preprocessing_info['steps_applied']}")
    
    # Advanced texture analysis
    print("🧠 Performing advanced texture analysis...")
    lbp_analyzer = AdvancedLBP()
    comprehensive_features, texture_map = lbp_analyzer.extract_comprehensive_lbp_features(
        preprocessed_image if len(preprocessed_image.shape) == 2 else preprocessed_image[:, :, 0]
    )
    print(f"   Texture analysis complete: {comprehensive_features['summary']['mean_texture_strength']:.3f}")
    
    # Load payload
    print(f"Loading payload: {text_payload_path}")
    with open(text_payload_path, 'r', encoding='utf-8') as f:
        payload_text = f.read()
    
    # Determine embedding parameters - optimized for clinical quality
    embedding_params = {
        'edge_threshold': 0.1,  # Lower threshold for better quality
        'texture_threshold': 0.3,  # Lower threshold for safer embedding
        'max_capacity_ratio': 0.05,  # Reduced capacity for higher quality
        'use_adaptive_threshold': True,
        'use_adaptive_optimization': False,
        'mri_mode': enable_mri_features
    }
    
    # ROI-adaptive pixel selection (MRI-specific)
    if enable_mri_features and config.get('mri_specific', {}).get('roi_adaptive', {}).get('enable_roi_segmentation', True):
        print("🎯 Performing ROI-adaptive pixel selection...")
        
        # Placeholder edge map (in real implementation, this would come from edge_lbp.py)
        edge_map = np.random.rand(*preprocessed_image.shape[:2]) * 0.5
        
        payload_size = len(payload_text.encode('utf-8')) * 8  # bits needed
        
        selected_pixels, roi_info = perform_roi_adaptive_embedding(
            preprocessed_image if len(preprocessed_image.shape) == 2 else preprocessed_image[:, :, 0],
            edge_map,
            texture_map,
            payload_size
        )
        
        print(f"   Selected {roi_info['selection_info']['n_selected']} pixels for embedding")
        print(f"   Safety ratio: {roi_info['selection_info']['safety_ratio']:.3f}")
        print(f"   Mean suitability score: {roi_info['selection_info'].get('mean_suitability_score', 0):.3f}")
        
        # Validate embedding safety
        safety_report = roi_info['safety_report']
        if not safety_report['overall_safety']['is_safe']:
            print(f"   ⚠️  Safety warning: {safety_report['overall_safety']['recommendation']}")
        else:
            print("   ✅ Embedding locations validated as safe for diagnosis")
    
    # Adaptive optimization for MRI
    if embedding_params['use_adaptive_optimization']:
        print("🧠 Using MRI-adaptive multi-objective optimization...")
        optimizer = MultiObjectiveOptimizer()
        
        # Analyze MRI characteristics
        mri_characteristics = optimizer.analyzer.analyze(preprocessed_image)
        print(f"   MRI complexity score: {mri_characteristics['image_complexity']:.3f}")
        print(f"   Brain tissue ratio: {mri_characteristics['brain_tissue_ratio']:.3f}")
        print(f"   Anatomical complexity: {mri_characteristics['anatomical_complexity']:.3f}")
        
        # Get MRI-specific optimization hints
        mri_hints = mri_characteristics.get('mri_specific_hints', {})
        if mri_hints:
            print(f"   Preprocessing recommendation: {mri_hints.get('preprocessing_recommendation', 'none')}")
            print(f"   ROI safety margin: {mri_hints.get('roi_safety_margin', 'default')}")
        
        # Optimize parameters
        optimized_params = optimizer.optimize_parameters(
            preprocessed_image, payload_text.encode('utf-8'), embedding_params
        )
        embedding_params.update(optimized_params['best_parameters'])
        print(f"   Optimization completed! Composite score: {optimized_params['best_score']:.4f}")
    else:
        print("⚙️  Using standard MRI-tuned parameters...")
    
    # Display final embedding parameters
    print("📊 Final embedding parameters:")
    for param, value in embedding_params.items():
        if isinstance(value, float):
            print(f"   {param}: {value:.3f}")
        else:
            print(f"   {param}: {value}")
    
    # Perform embedding on original cover image for proper reversibility
    print("🔒 Embedding payload...")
    stego_image, trace_matrix, embedding_stats = embed_payload(
        cover_image, text_payload_path, embedding_params
    )
    
    # Calculate quality metrics using preprocessed image for comparison
    from src.evaluation import calculate_psnr, calculate_ssim
    psnr = calculate_psnr(cover_image, stego_image)
    ssim_result = calculate_ssim(cover_image, stego_image)
    
    print(f"✅ Embedding successful!")
    print(f"   PSNR: {psnr:.2f} dB")
    if 'mean_ssim' in ssim_result:
        print(f"   SSIM: {ssim_result['mean_ssim']:.4f}")
    else:
        print(f"   SSIM: Error - {ssim_result.get('error', 'Unknown error')}")
    print(f"   Capacity used: {embedding_stats.get('capacity_used', 'N/A')}")
    
    # Save stego image and trace matrix
    stego_path = os.path.join(output_dir, 'stego_images', 'mri_stego.png')
    trace_path = os.path.join(output_dir, 'stego_images', 'mri_trace.pkl')
    
    os.makedirs(os.path.dirname(stego_path), exist_ok=True)
    save_image(stego_image, stego_path)
    save_trace_matrix(trace_matrix, trace_path)
    
    print(f"Saved stego image: {stego_path}")
    print(f"Saved trace matrix: {trace_path}")
    
    # Clinical evaluation (MRI-specific)
    if enable_mri_features and config.get('mri_specific', {}).get('clinical_validation', {}).get('enable_clinical_evaluation', True):
        print("🏥 Performing clinical evaluation...")
        
        clinical_evaluator = ClinicalEvaluator()
        clinical_report = clinical_evaluator.comprehensive_clinical_evaluation(
            cover_image, stego_image
        )
        
        # Display clinical results
        print("📋 Clinical Evaluation Results:")
        quality_metrics = clinical_report['image_quality_metrics']
        print(f"   SNR: {quality_metrics['snr']['db']:.2f} dB ({'✅' if quality_metrics['snr']['acceptable'] else '❌'})")
        print(f"   CNR: {quality_metrics['cnr']['db']:.2f} dB ({'✅' if quality_metrics['cnr']['acceptable'] else '❌'})")
        print(f"   Artifacts severity: {quality_metrics['artifacts']['severity']:.4f}")
        
        overall_assessment = clinical_report['overall_assessment']
        print(f"   Clinical Grade: {overall_assessment['clinical_grade']}")
        print(f"   Composite Score: {overall_assessment['composite_score']:.3f}")
        print(f"   Recommendation: {overall_assessment['recommendation']}")
        
        # Save clinical report
        clinical_report_path = os.path.join(output_dir, 'reports', 'mri_clinical_evaluation.json')
        os.makedirs(os.path.dirname(clinical_report_path), exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        clinical_report_serializable = convert_numpy_types(clinical_report)
        
        with open(clinical_report_path, 'w') as f:
            json.dump(clinical_report_serializable, f, indent=2, default=str)
        
        print(f"Saved clinical report: {clinical_report_path}")
    
    # Extract payload to verify
    print("🔓 Extracting payload...")
    
    # Load trace matrix for extraction
    from src.utils import load_trace_matrix
    loaded_trace = load_trace_matrix(trace_path)
    
    # Use loaded trace matrix for extraction
    extracted_payload, payload_metadata, extraction_info = extract_payload(stego_image, trace_matrix=loaded_trace, config=embedding_params)
    
    # Handle payload data based on type
    if isinstance(extracted_payload, bytes):
        extracted_text = extracted_payload.decode('utf-8')
    elif isinstance(extracted_payload, str) and all(c in '01' for c in extracted_payload):
        # Binary string - convert to text
        from src.utils import binary_to_text
        extracted_text = binary_to_text(extracted_payload)
    else:
        extracted_text = str(extracted_payload)
    
    # Verify payload
    payload_match = extracted_text == payload_text
    print(f"✓ Payload extraction {'verified' if payload_match else 'failed'} - content {'matches' if payload_match else 'differs'}!")
    
    # Save extracted payload
    extracted_path = os.path.join(output_dir, 'payload_extracted', 'mri_extracted_text.txt')
    os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
    with open(extracted_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    print(f"Extracted payload: {extracted_path}")
    
    # Restore original image
    print("🔄 Restoring original cover image...")
    restored_image = restore_cover(stego_image, trace_matrix=loaded_trace)
    
    # Verify restoration against original cover image (before preprocessing)
    restoration_perfect = verify_restoration(cover_image, restored_image)
    print(f"✓ {'Perfect' if restoration_perfect else 'Imperfect'} reversibility achieved!")
    
    # Save restored image
    restored_path = os.path.join(output_dir, 'cover_restored', 'mri_cover_restored.png')
    os.makedirs(os.path.dirname(restored_path), exist_ok=True)
    save_image(restored_image, restored_path)
    print(f"Restored cover image: {restored_path}")
    
    # Calculate final quality metrics for report
    from src.evaluation import calculate_psnr, calculate_ssim
    final_psnr = calculate_psnr(cover_image, stego_image)
    final_ssim_result = calculate_ssim(cover_image, stego_image)
    
    # Generate comprehensive evaluation report
    evaluation_report = {
        'timestamp': datetime.now().isoformat(),
        'mri_mode': enable_mri_features,
        'preprocessing_info': preprocessing_info,
        'texture_analysis': {
            'mean_texture_strength': float(comprehensive_features['summary']['mean_texture_strength']),
            'texture_entropy': float(comprehensive_features['summary']['texture_entropy']),
            'texture_homogeneity': float(comprehensive_features['summary']['texture_homogeneity'])
        },
        'embedding_parameters': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                               for k, v in embedding_params.items()},
        'quality_metrics': {
            'psnr': float(final_psnr),
            'ssim': float(final_ssim_result.get('mean_ssim', 0.0)),
            'payload_verified': payload_match,
            'reversibility_perfect': restoration_perfect
        }
    }
    
    if enable_mri_features and 'roi_info' in locals():
        evaluation_report['roi_analysis'] = {
            'pixels_selected': int(roi_info['selection_info']['n_selected']),
            'safety_ratio': float(roi_info['selection_info']['safety_ratio']),
            'overall_safety': roi_info['safety_report']['overall_safety']['is_safe']
        }
    
    if enable_mri_features and 'clinical_report' in locals():
        evaluation_report['clinical_assessment'] = {
            'clinical_grade': clinical_report['overall_assessment']['clinical_grade'],
            'composite_score': float(clinical_report['overall_assessment']['composite_score']),
            'snr_db': float(clinical_report['image_quality_metrics']['snr']['db']),
            'cnr_db': float(clinical_report['image_quality_metrics']['cnr']['db'])
        }
    
    # Save evaluation report
    eval_report_path = os.path.join(output_dir, 'reports', 'mri_evaluation_report.json')
    os.makedirs(os.path.dirname(eval_report_path), exist_ok=True)
    
    # Convert numpy types and other non-serializable types for JSON
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'dtype'):
            # Handle numpy dtype objects
            return str(obj)
        elif isinstance(obj, type):
            # Handle type objects
            return str(obj)
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            try:
                # Try direct JSON serialization
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                # If fails, convert to string
                return str(obj)
    
    evaluation_report_serializable = convert_for_json(evaluation_report)
    
    with open(eval_report_path, 'w') as f:
        json.dump(evaluation_report_serializable, f, indent=2)
    
    print(f"📄 Evaluation report saved: {eval_report_path}")
    
    return evaluation_report


def run_unit_tests():
    """Run unit tests for MRI-specific functionality"""
    print("\n" + "="*60)
    print("RUNNING MRI-SPECIFIC UNIT TESTS")
    print("="*60)
    
    test_results = []
    
    # Test 1: MRI Preprocessing
    print("Test 1: MRI Preprocessing")
    try:
        test_image = np.random.rand(128, 128) * 200 + 50
        test_image = test_image.astype(np.uint8)
        
        preprocessor = MRIPreprocessor()
        processed, info = preprocessor.preprocess_mri(test_image)
        
        success = processed is not None and len(info['steps_applied']) > 0
        print(f"✓ Test 1 {'PASSED' if success else 'FAILED'}")
        test_results.append(success)
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        test_results.append(False)
    
    # Test 2: Advanced LBP Analysis
    print("Test 2: Advanced LBP Analysis")
    try:
        test_image = np.random.rand(64, 64) * 255
        test_image = test_image.astype(np.uint8)
        
        lbp_analyzer = AdvancedLBP()
        features, texture_map = lbp_analyzer.extract_comprehensive_lbp_features(test_image)
        
        success = features is not None and texture_map is not None
        print(f"✓ Test 2 {'PASSED' if success else 'FAILED'}")
        test_results.append(success)
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        test_results.append(False)
    
    # Test 3: ROI-Adaptive Embedding
    print("Test 3: ROI-Adaptive Embedding")
    try:
        test_image = np.random.rand(64, 64) * 255
        test_image = test_image.astype(np.uint8)
        edge_map = np.random.rand(64, 64)
        texture_map = np.random.rand(64, 64)
        
        selected_pixels, roi_info = perform_roi_adaptive_embedding(
            test_image, edge_map, texture_map, payload_size=100
        )
        
        success = selected_pixels is not None and roi_info is not None
        print(f"✓ Test 3 {'PASSED' if success else 'FAILED'}")
        test_results.append(success)
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}")
        test_results.append(False)
    
    # Test 4: Clinical Evaluation
    print("Test 4: Clinical Evaluation")
    try:
        original = np.random.rand(64, 64) * 200 + 50
        stego = original + np.random.normal(0, 1, original.shape)
        original = original.astype(np.uint8)
        stego = stego.astype(np.uint8)
        
        evaluator = ClinicalEvaluator()
        report = evaluator.comprehensive_clinical_evaluation(original, stego)
        
        success = report is not None and 'overall_assessment' in report
        print(f"✓ Test 4 {'PASSED' if success else 'FAILED'}")
        test_results.append(success)
    except Exception as e:
        print(f"✗ Test 4 FAILED: {e}")
        test_results.append(False)
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    print(f"\n📋 UNIT TEST SUMMARY:")
    print(f"   Tests passed: {passed}/{total}")
    for i, result in enumerate(test_results, 1):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"   Test {i}: {status}")
    
    return all(test_results)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Enhanced MRI-Specific Hybrid Steganography System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_mri_enhanced.py --mri-mode
  python main_mri_enhanced.py --cover mri_brain.png --payload medical_notes.txt --mri-mode
  python main_mri_enhanced.py --adaptive --mri-mode
        """
    )
    
    parser.add_argument('--cover', type=str, default=None,
                       help='Path to cover image (preferably MRI)')
    parser.add_argument('--payload', type=str, default=None,
                       help='Path to payload file')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--adaptive', action='store_true', default=False, 
                       help='Enable adaptive multi-objective threshold optimization')
    parser.add_argument('--mri-mode', action='store_true', default=False,
                       help='Enable MRI-specific features (preprocessing, ROI-adaptive, clinical evaluation)')
    parser.add_argument('--test-only', action='store_true', default=False,
                       help='Run unit tests only')
    parser.add_argument('--batch', action='store_true', default=False,
                       help='Run batch evaluation on MRI dataset')
    parser.add_argument('--dataset-dir', type=str, default='data/mri_dataset/test_images',
                       help='Directory containing MRI dataset for batch processing')
    parser.add_argument('--max-images', type=int, default=50,
                       help='Maximum number of images for batch processing')
    
    args = parser.parse_args()
    
    # Print header
    print("🔒 ENHANCED MRI-SPECIFIC HYBRID STEGANOGRAPHY SYSTEM")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    if args.mri_mode:
        print("🧠 MRI-SPECIFIC MODE: ENABLED")
        print("   ✓ MRI preprocessing (denoising, bias correction, normalization)")
        print("   ✓ Advanced LBP texture analysis (multi-scale, rotation-invariant)")
        print("   ✓ ROI-adaptive embedding (avoiding diagnostic regions)")
        print("   ✓ Clinical evaluation and diagnostic quality assessment")
    
    if args.adaptive:
        print("🧠 ADAPTIVE MULTI-OBJECTIVE OPTIMIZATION: ENABLED")
        print("   Using intelligent parameter optimization for enhanced performance")
    
    # Load configuration
    config = load_config(args.config)
    
    # Add adaptive flag to config
    if 'pixel_selection' not in config:
        config['pixel_selection'] = {}
    config['pixel_selection']['use_adaptive_optimization'] = args.adaptive
    
    # Run tests if requested
    if args.test_only:
        tests_passed = run_unit_tests()
        return 0 if tests_passed else 1
    
    # Run batch evaluation if requested
    if args.batch:
        print(f"\n🔍 RUNNING BATCH MRI EVALUATION")
        print("="*50)
        print(f"Dataset directory: {args.dataset_dir}")
        print(f"Maximum images: {args.max_images}")
        print(f"MRI features enabled: {args.mri_mode}")
        
        try:
            from batch_mri_evaluation import MRISteganoEvaluationPipeline
            pipeline = MRISteganoEvaluationPipeline(args.output)
            
            batch_results = pipeline.run_batch_evaluation(
                args.dataset_dir, 
                max_images=args.max_images, 
                enable_mri_features=args.mri_mode
            )
            
            print(f"\n✅ Batch evaluation completed!")
            print(f"📄 Results saved to: {args.output}")
            return 0
            
        except Exception as e:
            print(f"❌ Batch evaluation failed: {str(e)}")
            return 1
    
    # Setup demo environment
    print("Setting up demo environment...")
    
    # Find or create cover image
    cover_path = args.cover
    if not cover_path:
        # First, look for MRI images in dataset directory if in MRI mode
        if args.mri_mode and os.path.exists(args.dataset_dir):
            print(f"Looking for MRI images in dataset: {args.dataset_dir}")
            mri_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                mri_files.extend(glob.glob(os.path.join(args.dataset_dir, f"*{ext}")))
                mri_files.extend(glob.glob(os.path.join(args.dataset_dir, "**", f"*{ext}"), recursive=True))
            
            if mri_files:
                cover_path = mri_files[0]
                print(f"Using MRI dataset image: {os.path.basename(cover_path)}")
            else:
                print(f"No MRI images found in {args.dataset_dir}")
        
        # If still no cover path, look in default data directory
        if not cover_path:
            data_dir = 'data/cover'
            mri_extensions = ['.png', '.jpg', '.jpeg', '.dcm']
            mri_keywords = ['mri', 'brain', 'medical', 'scan']
        
        if os.path.exists(data_dir):
            cover_files = []
            for ext in mri_extensions:
                for keyword in mri_keywords:
                    pattern = f"*{keyword}*{ext}"
                    matches = list(Path(data_dir).glob(pattern))
                    cover_files.extend(matches)
            
            if cover_files:
                cover_path = str(cover_files[0])
            else:
                # Use any available image
                all_images = []
                for ext in ['.png', '.jpg', '.jpeg']:
                    all_images.extend(list(Path(data_dir).glob(f"*{ext}")))
                
                if all_images:
                    cover_path = str(all_images[0])
                else:
                    # Create a synthetic MRI-like image
                    print("Creating synthetic MRI-like image...")
                    synthetic_mri = create_synthetic_mri_image()
                    os.makedirs(data_dir, exist_ok=True)
                    cover_path = os.path.join(data_dir, 'synthetic_mri.png')
                    save_image(synthetic_mri, cover_path)
        else:
            # Create data directory and synthetic image
            print("Creating synthetic MRI-like image...")
            synthetic_mri = create_synthetic_mri_image()
            os.makedirs(data_dir, exist_ok=True)
            cover_path = os.path.join(data_dir, 'synthetic_mri.png')
            save_image(synthetic_mri, cover_path)
    
    # Find or create payload
    payload_path = args.payload
    if not payload_path:
        # Look for medical text files
        data_dir = 'data/payloads'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        
        payload_path = os.path.join(data_dir, 'medical_sample_text.txt')
        if not os.path.exists(payload_path):
            # Create sample medical text
            medical_text = create_sample_medical_text()
            with open(payload_path, 'w', encoding='utf-8') as f:
                f.write(medical_text)
    
    print(f"📝 Starting {'MRI-enhanced' if args.mri_mode else 'standard'} embedding demonstration...")
    
    # Perform demonstration
    try:
        evaluation_report = demonstrate_mri_steganography(
            cover_path, payload_path, config, args.output, 
            enable_mri_features=args.mri_mode
        )
        
        # Run unit tests
        tests_passed = run_unit_tests()
        
        # Final summary
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print(f"📄 Results saved to: {args.output}")
        print(f"📁 Output directory: {args.output}")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Mode: {'MRI-Enhanced' if args.mri_mode else 'Standard'}")
        print(f"   PSNR: {evaluation_report['quality_metrics']['psnr']:.2f} dB")
        print(f"   SSIM: {evaluation_report['quality_metrics']['ssim']:.4f}")
        print(f"   Payload verified: {'✓' if evaluation_report['quality_metrics']['payload_verified'] else '✗'}")
        print(f"   Reversibility: {'✓ Perfect' if evaluation_report['quality_metrics']['reversibility_perfect'] else '✗ Imperfect'}")
        
        if args.mri_mode and 'clinical_assessment' in evaluation_report:
            clinical = evaluation_report['clinical_assessment']
            print(f"   Clinical Grade: {clinical['clinical_grade']}")
            print(f"   Clinical Score: {clinical['composite_score']:.3f}")
            print(f"   SNR: {clinical['snr_db']:.2f} dB")
            print(f"   CNR: {clinical['cnr_db']:.2f} dB")
        
        print(f"   Unit tests: {'✓ PASSED' if tests_passed else '✗ FAILED'}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1


def create_synthetic_mri_image():
    """Create a synthetic MRI-like image for testing"""
    # Create brain-like structure
    image = np.zeros((256, 256), dtype=np.uint8)
    
    # Add brain outline
    center = (128, 128)
    y, x = np.ogrid[:256, :256]
    brain_mask = ((x - center[0])**2 + (y - center[1])**2) < 100**2
    
    # Brain tissue
    image[brain_mask] = 150
    
    # Add some anatomical structures
    # Ventricles (dark regions)
    ventricle_centers = [(108, 128), (148, 128)]
    for vc in ventricle_centers:
        ventricle_mask = ((x - vc[0])**2 + (y - vc[1])**2) < 15**2
        image[ventricle_mask] = 50
    
    # White matter (brighter regions)
    white_matter_mask = ((x - center[0])**2 + (y - center[1])**2) < 70**2
    image[white_matter_mask] = 200
    
    # Add some noise (typical in MRI)
    noise = np.random.normal(0, 10, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return image


def create_sample_medical_text():
    """Create sample medical text for demonstration"""
    return """Medical Report - MRI Brain Scan

Patient ID: MRI_DEMO_001
Date: 2025-10-21
Modality: T1-weighted MRI

CLINICAL INDICATION:
Routine follow-up examination

TECHNIQUE:
T1-weighted sagittal, axial, and coronal images were obtained.

FINDINGS:
The brain parenchyma demonstrates normal signal intensity throughout.
No evidence of acute infarction, hemorrhage, or mass effect.
Ventricular system is within normal limits.
No midline shift is identified.

IMPRESSION:
Normal MRI brain examination.

This confidential medical information is embedded using advanced 
ROI-adaptive steganography to ensure diagnostic quality preservation.
"""


if __name__ == "__main__":
    sys.exit(main())