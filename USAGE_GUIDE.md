# MRI Steganography - Usage Guide

## Setup untuk Gambar MRI 512x512 JPG

### 1. Struktur Direktori

Tempatkan gambar MRI Anda dalam struktur direktori berikut:

```
data/
├── mri_dataset/
│   ├── test_images/          # Gambar MRI untuk testing umum
│   │   ├── mri_001.jpg
│   │   ├── mri_002.jpg
│   │   └── ...
│   └── validation_set/       # 50 gambar MRI untuk validasi radiologist
│       ├── validation_001.jpg
│       ├── validation_002.jpg
│       └── ...
└── payloads/
    └── medical_sample_text.txt
```

### 2. Format Gambar yang Didukung

- **Format**: JPG/JPEG (recommended untuk MRI)
- **Ukuran**: 512x512 pixels
- **Bit Depth**: 8-bit grayscale atau RGB
- **Naming**: Sequential naming (mri_001.jpg, validation_001.jpg, etc.)

## Penggunaan

### 1. Single Image Evaluation

```bash
# Evaluasi gambar MRI tunggal dengan fitur MRI
python main_mri_enhanced.py --cover data/mri_dataset/test_images/mri_001.jpg --mri-mode

# Dengan payload khusus
python main_mri_enhanced.py --cover mri_brain.jpg --payload medical_notes.txt --mri-mode
```

### 2. Batch Evaluation (Multiple Images)

```bash
# Evaluasi batch dengan fitur MRI
python main_mri_enhanced.py --batch --dataset-dir data/mri_dataset/test_images/ --mri-mode --max-images 50

# Atau menggunakan script khusus batch
python batch_mri_evaluation.py --dataset_dir data/mri_dataset/test_images/ --mri_features --max_images 50
```

### 3. Radiologist Validation Setup

```bash
# Setup framework validasi radiologist
python radiologist_validation.py --setup --validation_dir data/mri_dataset/validation_set/

# Process validation images untuk perbandingan
python radiologist_validation.py --setup --process --validation_dir data/mri_dataset/validation_set/

# Analisis hasil validasi radiologist
python radiologist_validation.py --analyze --results_file validation_results.json
```

## Comprehensive Evaluation Metrics

### Metrik yang Dihitung:

#### 1. Image Quality Metrics
- **PSNR** (Peak Signal-to-Noise Ratio): Mengukur kualitas gambar
- **SSIM** (Structural Similarity Index): Mengukur kesamaan struktural
- **MSE** (Mean Squared Error): Error rata-rata kuadrat
- **RMSE** (Root Mean Squared Error): Akar error rata-rata kuadrat
- **MAE** (Mean Absolute Error): Error rata-rata absolut
- **NCC** (Normalized Cross-Correlation): Korelasi ternormalisasi
- **UIQI** (Universal Image Quality Index): Indeks kualitas universal

#### 2. Security Metrics
- **UACI** (Unified Average Changing Intensity): Perubahan intensitas rata-rata
- **NPCR** (Number of Pixels Change Rate): Persentase pixel yang berubah
- **Correlation Analysis**: Analisis korelasi pixel
- **Histogram Analysis**: Analisis distribusi histogram
- **Entropy Analysis**: Analisis entropi informasi

#### 3. Performance Metrics
- **Embedding Capacity**: Kapasitas embedding (bits per pixel)
- **Extraction Accuracy**: Akurasi ekstraksi payload
- **Runtime Performance**: Waktu embedding dan ekstraksi
- **Throughput**: Bits per second processing

#### 4. Clinical Validation Metrics
- **Diagnostic Accuracy**: Akurasi diagnosis sebelum vs sesudah embedding
- **Visual Grading Analysis**: Penilaian kualitas visual oleh radiologist
- **Clinical Acceptability**: Persentase acceptability untuk penggunaan klinis
- **Statistical Significance**: Uji statistik (paired t-test, Wilcoxon)

### Runtime Measurement

System otomatis mengukur:
- **Embedding Time**: Waktu proses embedding
- **Extraction Time**: Waktu proses ekstraksi
- **Preprocessing Time**: Waktu preprocessing MRI
- **Total Processing Time**: Total waktu keseluruhan

## Radiologist Validation (50 Test Images)

### Protocol Validasi:

1. **Randomization**: Gambar dipresentasikan secara acak
2. **Blinding**: Radiologist tidak tahu mana gambar original vs processed
3. **Rating Scale**: 1-5 scale untuk kualitas diagnostik
4. **Criteria Evaluation**:
   - Overall image quality
   - Diagnostic confidence
   - Anatomical structure visibility
   - Lesion detectability
   - Noise perception
   - Artifact presence

### Setup Validasi:

```bash
# 1. Setup framework validasi
python radiologist_validation.py --setup --validation_dir data/mri_dataset/validation_set/

# 2. Process 50 gambar validasi
python radiologist_validation.py --setup --process --validation_dir data/mri_dataset/validation_set/

# 3. Berikan file Excel/JSON kepada radiologist untuk evaluasi
# File akan tersedia di: results/radiologist_validation/[study_name]/

# 4. Setelah radiologist selesai, analisis hasil
python radiologist_validation.py --analyze --results_file path/to/completed_validation.json
```

### Output Validasi:

- **Diagnostic Accuracy Comparison**: Original vs Stego accuracy (%)
- **Statistical Analysis**: p-values, effect sizes, confidence intervals
- **Agreement Analysis**: Inter-rater agreement, confusion matrices
- **Clinical Recommendations**: Berdasarkan hasil analisis
- **Visualizations**: Plots dan grafik perbandingan

## Output Results

### File Output yang Dihasilkan:

```
results/
├── stego_images/           # Gambar steganografi
├── payload_extracted/      # Payload yang diekstraksi
├── cover_restored/         # Cover image yang di-restore
├── reports/               # Laporan evaluasi komprehensif
│   ├── mri_evaluation_report.json
│   ├── comprehensive_report_[timestamp].json
│   └── evaluation_summary.txt
├── radiologist_validation/ # Hasil validasi radiologist
└── visualization/         # Grafik dan plot hasil
```

### Contoh Hasil Evaluasi:

```json
{
  "image_quality_metrics": {
    "psnr": 67.50,
    "ssim": 0.9998,
    "mse": 0.12,
    "uaci": 0.334,
    "npcr": 99.61
  },
  "performance_metrics": {
    "embedding_capacity_bpp": 0.024,
    "extraction_accuracy_byte": 1.0,
    "embedding_time_seconds": 0.145,
    "extraction_time_seconds": 0.089
  },
  "clinical_validation": {
    "original_diagnostic_accuracy": 98.0,
    "stego_diagnostic_accuracy": 96.0,
    "accuracy_preservation_percent": 97.96,
    "recommended_for_clinical_use": true
  }
}
```

## Best Practices

### 1. Persiapan Data MRI
- Gunakan gambar MRI berkualitas tinggi (512x512)
- Pastikan gambar sudah di-anonymize (hapus data pasien)
- Sertakan berbagai jenis sequence MRI (T1, T2, FLAIR, dll.)
- Gunakan format JPG untuk compatibility optimal

### 2. Parameter Configuration
- Gunakan `--mri-mode` untuk fitur MRI-specific
- Adjust `--max-images` sesuai kebutuhan computational
- Gunakan `--adaptive` untuk optimization otomatis

### 3. Clinical Validation
- Libatkan radiologist berpengalaman untuk validasi
- Gunakan minimal 50 gambar untuk validasi statistik
- Pastikan representasi yang baik dari berbagai kasus

### 4. Performance Optimization
- Gunakan virtual environment untuk dependency isolation
- Monitor RAM usage untuk batch processing besar
- Simpan intermediate results untuk debugging

## Troubleshooting

### Error Umum:

1. **"No MRI images found"**
   - Pastikan path direktori benar
   - Check format file (.jpg, .jpeg, .png)
   - Verifikasi permission read pada direktori

2. **"Insufficient safe pixels for embedding"**
   - ROI terlalu restrictive, adjust safety margin
   - Gunakan payload yang lebih kecil
   - Check texture analysis threshold

3. **Memory errors pada batch processing**
   - Reduce `--max-images`
   - Close other applications
   - Gunakan processing chunking

### Performance Tips:

- Gunakan SSD untuk I/O optimal
- Enable MRI preprocessing cache
- Monitor GPU usage jika tersedia
- Implement parallel processing untuk batch

## Citation

Jika menggunakan sistem ini untuk penelitian, mohon cite:

```
@article{mri_steganography_2025,
  title={Enhanced MRI-Specific Steganography with ROI-Adaptive Embedding and Clinical Validation},
  author={Research Team},
  journal={Medical Imaging and Information Security},
  year={2025}
}
```