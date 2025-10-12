#!/usr/bin/env python3
"""
Unit Tests for Hybrid Edge Detection + LBP + Reversible LSB Steganography
Tests the core functionality with specific test cases as requested
"""

import unittest
import numpy as np
import tempfile
import os
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import (
    text_to_binary, binary_to_text, create_sample_payloads,
    save_image, load_image, save_trace_matrix, load_trace_matrix
)
from src.embedding import embed_payload, extract_payload
from src.restore import restore_cover, verify_restoration
from src.evaluation import evaluate_metrics
from src.edge_lbp import preprocess_image, select_embedding_pixels


class TestSteganographyCore(unittest.TestCase):
    """Test core steganography functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test cover image with high texture
        np.random.seed(42)  # For reproducible tests
        self.test_cover = self.create_textured_image((256, 256))
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_textured_image(self, size):
        """Create a textured test image for better embedding"""
        # Start with noise
        image = np.random.randint(50, 200, size, dtype=np.uint8)
        
        # Add checkerboard pattern for edges
        block_size = 16
        for i in range(0, size[0], block_size):
            for j in range(0, size[1], block_size):
                if (i // block_size + j // block_size) % 2 == 0:
                    image[i:i+block_size, j:j+block_size] += 30
        
        # Ensure values are in valid range
        image = np.clip(image, 0, 255)
        return image
    
    def test_text_to_binary_conversion(self):
        """Test text to binary conversion and back"""
        test_text = "hello"
        binary = text_to_binary(test_text)
        recovered_text = binary_to_text(binary)
        
        self.assertEqual(test_text, recovered_text)
        self.assertEqual(len(binary), len(test_text) * 8)
    
    def test_hello_text_embedding(self):
        """Test embedding and extraction of 'hello' text as specified"""
        test_text = "hello"
        
        # Create temporary text file
        text_file = os.path.join(self.temp_dir, "hello.txt")
        with open(text_file, 'w') as f:
            f.write(test_text)
        
        # Configure embedding parameters
        config = {
            'edge_threshold': 0.3,
            'texture_threshold': 0.4,
            'max_capacity_ratio': 0.1,
            'use_adaptive_threshold': False,
            'add_metadata_header': True
        }
        
        # Test embedding
        stego_image, trace_matrix, embedding_info = embed_payload(
            self.test_cover, text_file, config
        )
        
        # Verify embedding info
        self.assertIsNotNone(stego_image)
        self.assertIsNotNone(trace_matrix)
        self.assertIsNotNone(embedding_info)
        self.assertEqual(stego_image.shape, self.test_cover.shape)
        
        # Test extraction
        extracted_file = os.path.join(self.temp_dir, "extracted_hello.txt")
        payload_binary, payload_metadata, extraction_info = extract_payload(
            stego_image, trace_matrix=trace_matrix, output_path=extracted_file
        )
        
        # Verify extraction
        self.assertTrue(os.path.exists(extracted_file))
        with open(extracted_file, 'r') as f:
            extracted_text = f.read()
        
        self.assertEqual(test_text, extracted_text)
        
        # Test reversibility
        restored_cover = restore_cover(stego_image, trace_matrix=trace_matrix)
        verification = verify_restoration(self.test_cover, restored_cover)
        
        self.assertTrue(verification['perfect_restoration'])
        self.assertEqual(verification['mse'], 0.0)
    
    def test_small_image_embedding(self):
        """Test embedding and extraction of small image as specified"""
        # Create small test image (32x32 checkerboard as specified)
        small_image = np.zeros((32, 32), dtype=np.uint8)
        for i in range(32):
            for j in range(32):
                if (i // 4 + j // 4) % 2 == 0:
                    small_image[i, j] = 255
        
        # Save test image
        image_file = os.path.join(self.temp_dir, "test_image.png")
        save_image(small_image, image_file)
        
        # Use larger cover image for image payload
        large_cover = self.create_textured_image((512, 512))
        
        # Configure embedding with higher capacity for image payload
        config = {
            'edge_threshold': 0.3,
            'texture_threshold': 0.4,
            'max_capacity_ratio': 0.15,  # Higher capacity for image
            'use_adaptive_threshold': False,
            'add_metadata_header': True
        }
        
        # Test embedding
        stego_image, trace_matrix, embedding_info = embed_payload(
            large_cover, image_file, config
        )
        
        # Verify embedding
        self.assertIsNotNone(stego_image)
        self.assertEqual(stego_image.shape, large_cover.shape)
        
        # Test extraction
        extracted_file = os.path.join(self.temp_dir, "extracted_image.png")
        payload_binary, payload_metadata, extraction_info = extract_payload(
            stego_image, trace_matrix=trace_matrix, output_path=extracted_file
        )
        
        # Verify extraction
        self.assertTrue(os.path.exists(extracted_file))
        extracted_image = load_image(extracted_file)
        
        # Convert to grayscale for comparison if needed
        if len(extracted_image.shape) == 3:
            extracted_image = extracted_image[:, :, 0]
        
        # Compare images
        np.testing.assert_array_equal(small_image, extracted_image)
        
        # Test reversibility
        restored_cover = restore_cover(stego_image, trace_matrix=trace_matrix)
        verification = verify_restoration(large_cover, restored_cover)
        
        self.assertTrue(verification['perfect_restoration'])
    
    def test_edge_detection_functionality(self):
        """Test edge detection and LBP feature extraction"""
        # Preprocess image
        processed = preprocess_image(self.test_cover)
        self.assertEqual(processed.shape, self.test_cover.shape)
        
        # Test pixel selection
        mask, edge_maps, texture_strength, coords = select_embedding_pixels(
            processed, edge_threshold=0.3, texture_threshold=0.4
        )
        
        # Verify results
        self.assertEqual(mask.shape, processed.shape)
        self.assertGreater(len(coords), 0)
        self.assertIn('canny', edge_maps)
        self.assertIn('sobel', edge_maps)
        self.assertIn('prewitt', edge_maps)
        self.assertIn('kirsch', edge_maps)
        self.assertIn('log', edge_maps)
    
    def test_trace_matrix_operations(self):
        """Test trace matrix save/load operations"""
        # Create test trace matrix
        test_coords = [(10, 20), (30, 40), (50, 60)]
        test_lsbs = [0, 1, 0]
        
        from src.utils import create_trace_matrix
        trace_matrix = create_trace_matrix(
            self.test_cover.shape, test_coords, test_lsbs
        )
        
        # Save and load trace matrix
        trace_file = os.path.join(self.temp_dir, "test_trace.pkl")
        save_trace_matrix(trace_matrix, trace_file)
        loaded_trace = load_trace_matrix(trace_file)
        
        # Verify integrity
        self.assertEqual(trace_matrix['image_shape'], loaded_trace['image_shape'])
        self.assertEqual(trace_matrix['embedding_coords'], loaded_trace['embedding_coords'])
        self.assertEqual(trace_matrix['original_lsbs'], loaded_trace['original_lsbs'])
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation"""
        # Create slightly modified image
        modified_image = self.test_cover.copy()
        modified_image[50:100, 50:100] = modified_image[50:100, 50:100] + 1
        modified_image = np.clip(modified_image, 0, 255)
        
        # Test evaluation
        report = evaluate_metrics(self.test_cover, modified_image)
        
        # Verify report structure
        self.assertIn('psnr', report)
        self.assertIn('ssim', report)
        self.assertIn('entropy', report)
        self.assertIn('histogram_similarity', report)
        self.assertIn('correlation', report)
        self.assertIn('quality_assessment', report)
        
        # Verify PSNR is reasonable
        self.assertGreater(report['psnr'], 20)  # Should be reasonable for small change
    
    def test_error_handling(self):
        """Test error handling for edge cases"""
        # Test with invalid payload
        with self.assertRaises(FileNotFoundError):
            embed_payload(self.test_cover, "nonexistent_file.txt")
        
        # Test with empty trace matrix
        with self.assertRaises(ValueError):
            restore_cover(self.test_cover, trace_matrix={})
        
        # Test with mismatched image shapes
        wrong_shape_image = np.zeros((100, 100), dtype=np.uint8)
        trace_matrix = {
            'image_shape': (256, 256),
            'embedding_coords': [(10, 20)],
            'original_lsbs': [0]
        }
        
        with self.assertRaises(ValueError):
            restore_cover(wrong_shape_image, trace_matrix=trace_matrix)
    
    def test_capacity_limits(self):
        """Test embedding capacity limits"""
        # Create large payload that should exceed capacity
        large_text = "A" * 10000  # Very large text
        text_file = os.path.join(self.temp_dir, "large_text.txt")
        with open(text_file, 'w') as f:
            f.write(large_text)
        
        config = {
            'edge_threshold': 0.3,
            'texture_threshold': 0.4,
            'max_capacity_ratio': 0.01,  # Very small capacity
            'use_adaptive_threshold': False,
            'add_metadata_header': True
        }
        
        # Should raise ValueError for payload too large
        with self.assertRaises(ValueError):
            embed_payload(self.test_cover, text_file, config)
    
    def test_different_image_types(self):
        """Test with different image types (grayscale vs color)"""
        # Test with color image
        color_cover = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # Create test payload
        test_text = "color_test"
        text_file = os.path.join(self.temp_dir, "color_test.txt")
        with open(text_file, 'w') as f:
            f.write(test_text)
        
        config = {
            'edge_threshold': 0.3,
            'texture_threshold': 0.4,
            'max_capacity_ratio': 0.1,
            'use_adaptive_threshold': False,
            'add_metadata_header': True
        }
        
        # Test embedding in color image
        stego_image, trace_matrix, _ = embed_payload(color_cover, text_file, config)
        
        # Verify it works with color images
        self.assertEqual(stego_image.shape, color_cover.shape)
        self.assertEqual(len(stego_image.shape), 3)
        
        # Test extraction and restoration
        extracted_file = os.path.join(self.temp_dir, "extracted_color_test.txt")
        extract_payload(stego_image, trace_matrix=trace_matrix, output_path=extracted_file)
        
        with open(extracted_file, 'r') as f:
            extracted_text = f.read()
        self.assertEqual(test_text, extracted_text)
        
        # Test restoration
        restored_cover = restore_cover(stego_image, trace_matrix=trace_matrix)
        verification = verify_restoration(color_cover, restored_cover)
        self.assertTrue(verification['perfect_restoration'])


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_sample_payload_creation(self):
        """Test creation of sample payloads"""
        text_path, image_path = create_sample_payloads(self.temp_dir)
        
        # Verify files exist
        self.assertTrue(os.path.exists(text_path))
        self.assertTrue(os.path.exists(image_path))
        
        # Verify content
        with open(text_path, 'r') as f:
            content = f.read()
        self.assertEqual(content, "hello")
        
        # Verify image
        image = load_image(image_path)
        self.assertEqual(image.shape, (32, 32))


def run_unit_tests():
    """Run all unit tests and return success status"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSteganographyCore))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Return True if all tests passed
    return result.wasSuccessful()


if __name__ == "__main__":
    print("🧪 Running Unit Tests for Hybrid Steganography System")
    print("=" * 60)
    
    success = run_unit_tests()
    
    if success:
        print("\n✅ All unit tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some unit tests failed!")
        sys.exit(1)