# Hybrid Edge Detection + LBP + Reversible LSB Steganography

A comprehensive steganography system that combines edge detection, Local Binary Pattern (LBP) texture analysis, and reversible LSB embedding for high-quality, imperceptible data hiding with 100% reversibility.

## 🌟 Features

- **Multi-Edge Detection**: Canny, Sobel, Prewitt, Kirsch, and Laplacian of Gaussian (LoG)
- **LBP Texture Analysis**: Local Binary Pattern for selecting high-texture regions
- **Reversible LSB Embedding**: 100% lossless restoration of original cover image
- **Adaptive Thresholds**: Automatic optimization for target PSNR and capacity
- **Comprehensive Evaluation**: PSNR, SSIM, entropy, histogram, correlation metrics
- **Multiple Payload Types**: Text and image payloads with auto-detection
- **Quality Assessment**: Automatic quality grading with recommendations

## 📁 Project Structure

```
hybrid_lsb_stego/
├── main.py                  # Main demo script
├── config.yaml              # Configuration parameters
├── requirements.txt         # Python dependencies
├── test_steganography.py    # Unit tests
├── src/
│   ├── __init__.py
│   ├── embedding.py         # Core embedding/extraction functions
│   ├── edge_lbp.py          # Edge detection and LBP analysis
│   ├── restore.py           # Reversible restoration functions
│   ├── evaluation.py        # Quality metrics and evaluation
│   └── utils.py             # Utility functions
├── data/
│   ├── cover/               # Cover images
│   ├── payloads/            # Payload files
│   └── results/
│       ├── stego_images/    # Generated stego images
│       ├── payload_extracted/ # Extracted payloads
│       ├── cover_restored/  # Restored cover images
│       └── reports/         # Evaluation reports
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or extract the project
cd hybrid_lsb_stego

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Demo

```bash
# Run complete demonstration with sample data
python main.py --setup-demo

# Run with custom cover image and payload
python main.py --cover path/to/cover.png --payload path/to/payload.txt

# Run unit tests only
python main.py --test
```

### 3. Basic Usage

```python
from src import embed_payload, extract_payload, restore_cover, evaluate_metrics

# Load cover image
cover_image = load_image('cover.png')

# Embed payload
stego_image, trace_matrix, info = embed_payload(cover_image, 'payload.txt')

# Extract payload
extract_payload(stego_image, trace_matrix=trace_matrix, output_path='extracted.txt')

# Restore original cover (100% reversible)
restored_cover = restore_cover(stego_image, trace_matrix=trace_matrix)

# Evaluate quality
report = evaluate_metrics(cover_image, stego_image, trace_matrix=trace_matrix)
```

## 🔧 Configuration

Edit `config.yaml` to customize parameters:

```yaml
# Edge detection thresholds
edge_detection:
  combination:
    threshold: 0.3

# LBP parameters
lbp:
  texture_threshold: 0.4

# Embedding capacity
pixel_selection:
  max_capacity_ratio: 0.1

# Quality targets
quality:
  target_psnr: 40.0
  min_ssim: 0.9
```

## 🧪 Testing

The system includes comprehensive unit tests:

```bash
# Run all tests
python test_steganography.py

# Run specific test methods
python -m unittest test_steganography.TestSteganographyCore.test_hello_text_embedding
```

### Test Cases

1. **Text Embedding**: "hello" text payload
2. **Image Embedding**: 32x32 checkerboard image
3. **Reversibility**: Perfect restoration verification
4. **Edge Cases**: Error handling and capacity limits

## 📊 Quality Metrics

The system evaluates multiple quality metrics:

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **Entropy**: Information content analysis
- **Histogram Similarity**: Statistical distribution comparison
- **Correlation**: Pixel-wise correlation analysis
- **Reversibility**: Perfect restoration verification

## 🎯 Applications

- **Medical Imaging**: Secure patient data embedding
- **Forensic Analysis**: Evidence authentication
- **E-Government**: Document integrity protection
- **Digital Rights Management**: Copyright protection

## 📈 Performance

Typical performance metrics:
- **PSNR**: 35-45 dB (excellent imperceptibility)
- **SSIM**: 0.95-0.98 (high structural similarity)
- **Capacity**: 5-10% of image pixels
- **Reversibility**: 100% perfect restoration

## 🔬 Algorithm Overview

1. **Preprocessing**: Convert to grayscale, normalize
2. **Edge Detection**: Apply 5 different edge detection algorithms
3. **LBP Analysis**: Extract texture features
4. **Pixel Selection**: Combine edge and texture maps
5. **LSB Embedding**: Reversible embedding with trace matrix
6. **Quality Assessment**: Comprehensive evaluation

## 📝 Output Files

After running the demo, you'll find:

- `stego_images/`: Generated steganographic images
- `payload_extracted/`: Extracted payload files
- `cover_restored/`: Restored original images
- `reports/`: JSON evaluation reports and plots

## 🛠️ Dependencies

- Python 3.7+
- NumPy
- OpenCV
- scikit-image
- Pillow
- PyYAML
- SciPy
- Matplotlib

## 📄 License

This project is for research and educational purposes. Please cite appropriately if used in academic work.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## 📞 Support

For questions or issues:
1. Check the configuration in `config.yaml`
2. Run unit tests to verify installation
3. Review error messages in console output
4. Check generated reports for detailed metrics

## 🏆 Acknowledgments

This implementation combines research from:
- Edge detection algorithms
- Local Binary Pattern analysis
- Reversible steganography techniques
- Image quality assessment metrics