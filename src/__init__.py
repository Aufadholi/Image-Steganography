"""
Hybrid Edge Detection + LBP + Reversible LSB Steganography Package

This package implements a comprehensive steganography system that combines:
1. Multi-edge detection (Canny, Sobel, Prewitt, Kirsch, LoG)
2. Local Binary Pattern (LBP) texture analysis
3. Reversible LSB embedding with trace matrix
4. Comprehensive quality evaluation metrics

Main modules:
- embedding: Core embedding and extraction functions
- edge_lbp: Edge detection and LBP feature extraction
- restore: Reversible cover image restoration
- evaluation: Quality metrics and assessment
- utils: Utility functions for I/O and data handling
"""

__version__ = "1.0.0"
__author__ = "Steganography Research Team"
__email__ = "steganography@research.edu"

# Import main functions for easy access
from .embedding import embed_payload, extract_payload, embed_text, extract_text
from .restore import restore_cover, verify_restoration
from .evaluation import evaluate_metrics, save_evaluation_report
from .utils import load_image, save_image, create_sample_payloads

__all__ = [
    'embed_payload',
    'extract_payload', 
    'embed_text',
    'extract_text',
    'restore_cover',
    'verify_restoration',
    'evaluate_metrics',
    'save_evaluation_report',
    'load_image',
    'save_image',
    'create_sample_payloads'
]