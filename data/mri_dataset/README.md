# MRI Dataset Directory Structure

## Directory Contents:

### `test_images/`
Place your 512x512 JPG MRI images here for general testing.
- Expected format: `.jpg` or `.jpeg`
- Expected size: 512x512 pixels
- Naming convention: `mri_001.jpg`, `mri_002.jpg`, etc.

### `validation_set/`
Place 50 MRI images here for radiologist validation.
- Expected format: `.jpg` or `.jpeg`
- Expected size: 512x512 pixels
- Naming convention: `validation_001.jpg` to `validation_050.jpg`

## Usage Examples:

1. **Single MRI Image Testing:**
   ```
   python main_mri_enhanced.py --mri_mode --cover_path data/mri_dataset/test_images/mri_001.jpg
   ```

2. **Batch Processing for Validation:**
   ```
   python batch_mri_evaluation.py --validation_dir data/mri_dataset/validation_set/
   ```

3. **Comprehensive Evaluation:**
   ```
   python comprehensive_mri_evaluation.py --dataset_dir data/mri_dataset/
   ```

## Expected Image Characteristics:
- **Modality**: MRI (T1, T2, FLAIR, etc.)
- **Format**: JPEG
- **Resolution**: 512x512 pixels
- **Bit Depth**: 8-bit grayscale or RGB
- **Content**: Brain MRI scans

## Notes:
- Ensure images are anonymized (no patient information)
- Images should represent various MRI sequences and anatomical views
- For validation set, include diverse pathological cases