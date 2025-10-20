#!/usr/bin/env python3
"""
Hybrid Edge Detection + LBP + Reversible LSB Steganography
Main Demo Script with Adaptive Multi-Objective Optimization

This script demonstrates the complete workflow:
1. Load cover image and payload
2. Embed payload using hybrid approach (with optional adaptive optimization)
3. Extract payload from stego image
4. Restore original cover image
5. Evaluate quality metrics and generate reports
"""

import os
import sys
import argparse
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


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using default configuration.")
        return {}


def demonstrate_text_embedding(cover_path, text_payload_path, config, output_dir):
    """Demonstrate text embedding workflow with optional adaptive optimization"""
    print("\n" + "="*60)
    print("DEMONSTRATING TEXT EMBEDDING")
    print("="*60)
    
    # Load cover image
    print(f"Loading cover image: {cover_path}")
    cover_image = load_image(cover_path)
    print(f"Cover image shape: {cover_image.shape}")
    
    # Configure embedding parameters
    embedding_config = {
        'edge_threshold': config.get('pixel_selection', {}).get('adaptive_threshold', {}).get('edge_threshold_range', [0.2, 0.6])[0],
        'texture_threshold': config.get('lbp', {}).get('texture_threshold', 0.4),
        'max_capacity_ratio': config.get('pixel_selection', {}).get('max_capacity_ratio', 0.1),
        'use_adaptive_threshold': config.get('pixel_selection', {}).get('adaptive_threshold', {}).get('enabled', True),
        'use_adaptive_optimization': config.get('pixel_selection', {}).get('use_adaptive_optimization', False),
        'target_psnr': config.get('quality', {}).get('target_psnr', 40.0),
        'add_metadata_header': config.get('embedding', {}).get('add_metadata_header', True)
    }
    
    try:
        # Embed payload
        print(f"\nEmbedding text payload: {text_payload_path}")
        stego_image, trace_matrix, embedding_info = embed_payload(
            cover_image, text_payload_path, embedding_config
        )
        
        print(f"Embedding successful!")
        psnr_value = embedding_info.get('psnr', None)
        if psnr_value is not None:
            print(f"PSNR: {psnr_value:.2f} dB")
        else:
            print("PSNR: N/A")
        print(f"Capacity used: {embedding_info.get('utilization_ratio', 0):.2%}")
        
        # Save stego image and trace matrix
        stego_path = os.path.join(output_dir, 'stego_images', 'text_stego.png')
        trace_path = os.path.join(output_dir, 'stego_images', 'text_trace.pkl')
        
        save_image(stego_image, stego_path)
        save_trace_matrix(trace_matrix, trace_path)
        print(f"Saved stego image: {stego_path}")
        print(f"Saved trace matrix: {trace_path}")
        
        # Extract payload
        print(f"\nExtracting payload from stego image...")
        extracted_path = os.path.join(output_dir, 'payload_extracted', 'extracted_text.txt')
        payload_binary, payload_metadata, extraction_info = extract_payload(
            stego_image, trace_matrix=trace_matrix, output_path=extracted_path
        )
        print(f"Extracted payload: {extracted_path}")
        
        # Verify extracted payload
        with open(text_payload_path, 'r', encoding='utf-8') as f:
            original_text = f.read().strip()
        with open(extracted_path, 'r', encoding='utf-8') as f:
            extracted_text = f.read().strip()
        
        if original_text == extracted_text:
            print("✓ Payload extraction verified - content matches!")
        else:
            print(f"✗ Payload extraction failed - content mismatch!")
            print(f"  Original: '{original_text}' (len={len(original_text)})")
            print(f"  Extracted: '{extracted_text}' (len={len(extracted_text)})")
        
        # Restore cover image
        print(f"\nRestoring original cover image...")
        restored_cover = restore_cover(stego_image, trace_matrix=trace_matrix)
        restored_path = os.path.join(output_dir, 'cover_restored', 'text_cover_restored.png')
        save_image(restored_cover, restored_path)
        print(f"Restored cover image: {restored_path}")
        
        # Verify restoration
        verification = verify_restoration(cover_image, restored_cover)
        if verification['perfect_restoration']:
            print("✓ Perfect reversibility achieved!")
        else:
            print(f"✗ Reversibility failed - MSE: {verification['mse']}")
        
        # Evaluate metrics
        print(f"\nEvaluating quality metrics...")
        evaluation_report = evaluate_metrics(
            cover_image, stego_image, text_payload_path, trace_matrix
        )
        
        # Save evaluation report
        report_path = os.path.join(output_dir, 'reports', 'text_evaluation.json')
        save_evaluation_report(evaluation_report, report_path)
        print(f"Evaluation report: {report_path}")
        
        # Create plots
        plots_dir = os.path.join(output_dir, 'reports', 'text_plots')
        create_evaluation_plots(cover_image, stego_image, plots_dir)
        print(f"Evaluation plots: {plots_dir}")
        
        # Print summary
        quality = evaluation_report.get('quality_assessment', {})
        print(f"\n📊 QUALITY SUMMARY:")
        print(f"   PSNR: {evaluation_report.get('psnr', 0):.2f} dB")
        print(f"   SSIM: {evaluation_report.get('ssim', {}).get('ssim', 0):.4f}")
        print(f"   Quality Level: {quality.get('quality_level', 'Unknown')}")
        print(f"   Imperceptibility Grade: {quality.get('imperceptibility_grade', 'Unknown')}")
        print(f"   Reversibility: {quality.get('reversibility_status', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Text embedding demonstration failed: {e}")
        return False


def run_unit_tests():
    """Run basic unit tests"""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    test_results = []
    
    # Test 1: Create and test with "hello" text
    try:
        print("Test 1: Text embedding with 'hello'")
        
        # Create test cover image
        test_cover = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        
        # Create temporary payload file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("hello")
            temp_payload_path = f.name
        
        try:
            # Test embedding
            stego_image, trace_matrix, embedding_info = embed_payload(
                test_cover, temp_payload_path, {
                    'edge_threshold': 0.3,
                    'texture_threshold': 0.4,
                    'max_capacity_ratio': 0.1,
                    'use_adaptive_threshold': False,
                    'use_adaptive_optimization': False,
                    'add_metadata_header': True
                }
            )
            
            # Test extraction
            payload_binary, payload_metadata, extraction_info = extract_payload(
                stego_image, trace_matrix=trace_matrix
            )
            
            # Test restoration
            restored_cover = restore_cover(stego_image, trace_matrix=trace_matrix)
            verification = verify_restoration(test_cover, restored_cover)
            
            if verification['perfect_restoration']:
                print("✓ Test 1 PASSED")
                test_results.append(("hello text test", True))
            else:
                print("✗ Test 1 FAILED - restoration not perfect")
                test_results.append(("hello text test", False))
                
        finally:
            os.unlink(temp_payload_path)
            
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        test_results.append(("hello text test", False))
    
    # Test 2: Small image embedding
    try:
        print("Test 2: Small image embedding")
        
        # Create test cover image
        test_cover = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        
        # Create small test image
        test_image = np.zeros((32, 32), dtype=np.uint8)
        for i in range(32):
            for j in range(32):
                if (i // 4 + j // 4) % 2 == 0:
                    test_image[i, j] = 255
        
        # Save temporary image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_image_path = f.name
        
        save_image(test_image, temp_image_path)
        
        try:
            # Test embedding
            stego_image, trace_matrix, embedding_info = embed_payload(
                test_cover, temp_image_path, {
                    'edge_threshold': 0.3,
                    'texture_threshold': 0.4,
                    'max_capacity_ratio': 0.15,  # Larger capacity for image
                    'use_adaptive_threshold': False,
                    'use_adaptive_optimization': False,
                    'add_metadata_header': True
                }
            )
            
            # Test restoration
            restored_cover = restore_cover(stego_image, trace_matrix=trace_matrix)
            verification = verify_restoration(test_cover, restored_cover)
            
            if verification['perfect_restoration']:
                print("✓ Test 2 PASSED")
                test_results.append(("small image test", True))
            else:
                print("✗ Test 2 FAILED - restoration not perfect")
                test_results.append(("small image test", False))
                
        finally:
            os.unlink(temp_image_path)
            
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        test_results.append(("small image test", False))
    
    # Print test summary
    print(f"\n📋 UNIT TEST SUMMARY:")
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    print(f"   Tests passed: {passed}/{total}")
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"   {test_name}: {status}")
    
    return all(result for _, result in test_results)


def setup_demo_environment(data_dir):
    """Set up the demo environment with sample data"""
    print("Setting up demo environment...")
    
    # Create sample payloads
    payloads_dir = os.path.join(data_dir, 'payloads')
    text_path, image_path = create_sample_payloads(payloads_dir)
    
    # Use existing cover images
    cover_dir = os.path.join(data_dir, 'cover')
    cover_files = [f for f in os.listdir(cover_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if cover_files:
        cover_path = os.path.join(cover_dir, cover_files[0])
    else:
        cover_path = os.path.join(cover_dir, 'default_cover.png')
        print("Creating default cover image...")
        # Create a simple default cover
        default_cover = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        save_image(default_cover, cover_path)
    
    return cover_path, text_path, image_path


def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(
        description="Hybrid Edge Detection + LBP + Reversible LSB Steganography Demo"
    )
    parser.add_argument('--cover', type=str, help='Path to cover image')
    parser.add_argument('--payload', type=str, help='Path to payload file')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--test', action='store_true', help='Run unit tests only')
    parser.add_argument('--setup-demo', action='store_true', help='Set up demo environment')
    parser.add_argument('--adaptive', action='store_true', default=False, 
                       help='Enable adaptive multi-objective threshold optimization')
    
    args = parser.parse_args()
    
    # Print header
    print("🔒 HYBRID EDGE DETECTION + LBP + REVERSIBLE LSB STEGANOGRAPHY")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    if args.adaptive:
        print("🧠 ADAPTIVE MULTI-OBJECTIVE OPTIMIZATION: ENABLED")
        print("   Using intelligent parameter optimization for enhanced performance")
    else:
        print("⚙️  Standard Mode: Using fixed configuration parameters")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Add adaptive flag to config
        if 'pixel_selection' not in config:
            config['pixel_selection'] = {}
        config['pixel_selection']['use_adaptive_optimization'] = args.adaptive
        
        # Run unit tests if requested
        if args.test:
            print("\n🧪 Running unit tests...")
            test_success = run_unit_tests()
            return 0 if test_success else 1
        
        # Set up demo environment if requested
        if args.setup_demo:
            data_dir = 'data'
            cover_path, text_path, image_path = setup_demo_environment(data_dir)
            print(f"Demo environment set up in '{data_dir}' directory")
            return 0
        
        # Determine cover and payload paths
        if args.cover and args.payload:
            cover_path = args.cover
            text_payload_path = args.payload
            image_payload_path = None
        else:
            # Use demo data
            data_dir = 'data'
            cover_path, text_payload_path, image_payload_path = setup_demo_environment(data_dir)
        
        # Create output directory
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        
        # Store configuration
        config_output_path = os.path.join(output_dir, 'config_used.yaml')
        with open(config_output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"\n📝 Starting {'adaptive' if args.adaptive else 'standard'} embedding demonstration...")
        
        # Run demonstrations
        results = {
            'timestamp': datetime.now().isoformat(),
            'adaptive_optimization': args.adaptive,
            'config_file': args.config,
            'demonstrations': []
        }
        
        # Text embedding demonstration
        text_success = demonstrate_text_embedding(cover_path, text_payload_path, config, output_dir)
        results['demonstrations'].append({
            'type': 'text',
            'cover': cover_path,
            'payload': text_payload_path,
            'success': text_success
        })
        
        # Run unit tests
        print("\n🧪 Running unit tests...")
        test_success = run_unit_tests()
        results['unit_tests_passed'] = test_success
        
        # Save results
        results_path = os.path.join(output_dir, 'demo_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 DEMONSTRATION COMPLETE!")
        print(f"📄 Results saved to: {results_path}")
        print(f"📁 Output directory: {output_dir}")
        
        # Print summary
        successful_demos = sum(1 for demo in results['demonstrations'] if demo['success'])
        total_demos = len(results['demonstrations'])
        
        print(f"\n📊 SUMMARY:")
        print(f"   Successful demonstrations: {successful_demos}/{total_demos}")
        print(f"   Unit tests: {'✓ PASSED' if test_success else '✗ FAILED'}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())