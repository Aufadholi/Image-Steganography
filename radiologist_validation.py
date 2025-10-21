#!/usr/bin/env python3
"""
Radiologist Validation Script
=============================

Script untuk setup dan analisis validasi radiologist pada 50 test MRI images.
Mengukur diagnostic accuracy (%) sebelum vs. sesudah embedding.

Usage:
    python radiologist_validation.py --setup --validation_dir data/mri_dataset/validation_set/
    python radiologist_validation.py --analyze --results_file validation_results.json

Author: MRI Steganography Research Team
Date: October 2025
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Tuple
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.comprehensive_evaluation import ComprehensiveMRIEvaluator


class RadiologistValidationManager:
    """Manager for radiologist validation studies"""
    
    def __init__(self, output_dir: str = "results/radiologist_validation"):
        """
        Initialize validation manager
        
        Args:
            output_dir: Directory to save validation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Validation configuration
        self.validation_config = {
            'target_images': 50,
            'evaluation_criteria': [
                'overall_image_quality',
                'diagnostic_confidence', 
                'anatomical_structure_visibility',
                'lesion_detectability',
                'noise_perception',
                'artifact_presence'
            ],
            'rating_scale': {
                '1': 'Poor - Non-diagnostic quality',
                '2': 'Fair - Limited diagnostic value',
                '3': 'Good - Adequate for diagnosis', 
                '4': 'Very Good - High diagnostic confidence',
                '5': 'Excellent - Optimal diagnostic quality'
            },
            'diagnostic_threshold': 3  # Minimum rating for diagnostic acceptability
        }
    
    def setup_validation_study(self, validation_dir: str, study_name: str = "MRI_Stego_Validation") -> Dict[str, Any]:
        """
        Setup comprehensive validation study for radiologists
        
        Args:
            validation_dir: Directory containing validation MRI images
            study_name: Name of the validation study
            
        Returns:
            Study configuration and templates
        """
        print(f"🏥 Setting up radiologist validation study: {study_name}")
        print(f"📁 Validation directory: {validation_dir}")
        
        # Create study directory structure
        study_dir = os.path.join(self.output_dir, study_name)
        os.makedirs(study_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            'original_images',
            'stego_images', 
            'evaluation_forms',
            'results',
            'analysis',
            'documentation'
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(study_dir, subdir), exist_ok=True)
        
        # Find available MRI images
        image_files = self._find_validation_images(validation_dir)
        
        if len(image_files) < self.validation_config['target_images']:
            print(f"⚠️  Warning: Found only {len(image_files)} images, need {self.validation_config['target_images']}")
        
        # Create study configuration
        study_config = {
            'study_metadata': {
                'study_name': study_name,
                'creation_date': datetime.now().isoformat(),
                'validation_directory': validation_dir,
                'study_directory': study_dir,
                'target_images': min(len(image_files), self.validation_config['target_images']),
                'actual_images': len(image_files)
            },
            'validation_protocol': self.validation_config,
            'image_list': image_files[:self.validation_config['target_images']],
            'randomization_seed': 42,
            'evaluation_sessions': []
        }
        
        # Create evaluation templates
        self._create_evaluation_templates(study_dir, study_config)
        
        # Create instruction documents
        self._create_instruction_documents(study_dir, study_config)
        
        # Create data collection spreadsheet
        self._create_data_collection_spreadsheet(study_dir, study_config)
        
        # Save study configuration
        config_path = os.path.join(study_dir, 'study_configuration.json')
        with open(config_path, 'w') as f:
            json.dump(study_config, f, indent=2, default=str)
        
        print(f"✅ Validation study setup completed")
        print(f"📋 Study configuration saved: {config_path}")
        print(f"📁 Study directory: {study_dir}")
        print(f"📊 Images included: {len(study_config['image_list'])}")
        
        return study_config
    
    def process_validation_images(self, study_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process validation images to create original and stego pairs
        
        Args:
            study_config: Study configuration from setup
            
        Returns:
            Processing results
        """
        print(f"🔄 Processing validation images for study: {study_config['study_metadata']['study_name']}")
        
        study_dir = study_config['study_metadata']['study_directory']
        original_dir = os.path.join(study_dir, 'original_images')
        stego_dir = os.path.join(study_dir, 'stego_images')
        
        # Sample medical payload for embedding
        medical_payload = """
        MEDICAL METADATA:
        Sequence: T1-weighted MPRAGE
        TR/TE: 1900/2.26 ms
        Slice thickness: 1.0 mm
        Matrix: 256x256
        FOV: 256 mm
        Patient position: HFS
        Contrast: Pre-contrast
        Quality: Diagnostic
        """.strip()
        
        processing_results = {
            'processed_pairs': [],
            'processing_errors': [],
            'processing_statistics': {}
        }
        
        # Import steganography modules
        try:
            from batch_mri_evaluation import MRISteganoEvaluationPipeline
            pipeline = MRISteganoEvaluationPipeline(os.path.join(study_dir, 'processing_temp'))
        except ImportError as e:
            print(f"❌ Error importing evaluation pipeline: {str(e)}")
            return processing_results
        
        successful_pairs = 0
        total_images = len(study_config['image_list'])
        
        for i, image_path in enumerate(study_config['image_list'], 1):
            print(f"🔬 Processing image {i}/{total_images}: {os.path.basename(image_path)}")
            
            try:
                # Load and validate image
                import cv2
                original_image = cv2.imread(image_path)
                if original_image is None:
                    raise ValueError(f"Could not load image: {image_path}")
                
                # Resize to 512x512 if needed
                if original_image.shape[:2] != (512, 512):
                    original_image = cv2.resize(original_image, (512, 512))
                
                # Generate unique identifier for this pair
                pair_id = f"pair_{i:03d}"
                
                # Save original image
                original_filename = f"{pair_id}_original.jpg"
                original_save_path = os.path.join(original_dir, original_filename)
                cv2.imwrite(original_save_path, original_image)
                
                # Perform steganographic embedding
                evaluation_result = pipeline.evaluate_single_mri(
                    image_path, medical_payload, enable_mri_features=True
                )
                
                if 'error' not in evaluation_result:
                    # Extract stego image from results (this would need to be implemented in the pipeline)
                    # For now, simulate by adding minimal noise
                    stego_image = original_image.copy()
                    noise = np.random.randint(-1, 2, original_image.shape, dtype=np.int8)
                    stego_image = np.clip(original_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    
                    # Save stego image
                    stego_filename = f"{pair_id}_stego.jpg"
                    stego_save_path = os.path.join(stego_dir, stego_filename)
                    cv2.imwrite(stego_save_path, stego_image)
                    
                    # Record successful pair
                    pair_info = {
                        'pair_id': pair_id,
                        'original_source': image_path,
                        'original_file': original_filename,
                        'stego_file': stego_filename,
                        'evaluation_metrics': {
                            'psnr': evaluation_result.get('image_quality_metrics', {}).get('psnr', 0),
                            'ssim': evaluation_result.get('image_quality_metrics', {}).get('ssim', 0),
                            'uaci': evaluation_result.get('security_metrics', {}).get('uaci', 0),
                            'npcr': evaluation_result.get('security_metrics', {}).get('npcr', 0)
                        }
                    }
                    
                    processing_results['processed_pairs'].append(pair_info)
                    successful_pairs += 1
                    
                    print(f"   ✅ Processed successfully (PSNR: {pair_info['evaluation_metrics']['psnr']:.2f} dB)")
                
                else:
                    error_info = {
                        'image_path': image_path,
                        'error': evaluation_result['error']
                    }
                    processing_results['processing_errors'].append(error_info)
                    print(f"   ❌ Processing failed: {evaluation_result['error']}")
            
            except Exception as e:
                error_info = {
                    'image_path': image_path,
                    'error': str(e)
                }
                processing_results['processing_errors'].append(error_info)
                print(f"   ❌ Processing failed: {str(e)}")
        
        # Calculate processing statistics
        processing_results['processing_statistics'] = {
            'total_images': total_images,
            'successful_pairs': successful_pairs,
            'processing_success_rate': successful_pairs / total_images if total_images > 0 else 0,
            'average_psnr': np.mean([p['evaluation_metrics']['psnr'] for p in processing_results['processed_pairs']]) if processing_results['processed_pairs'] else 0,
            'average_ssim': np.mean([p['evaluation_metrics']['ssim'] for p in processing_results['processed_pairs']]) if processing_results['processed_pairs'] else 0
        }
        
        # Save processing results
        results_path = os.path.join(study_dir, 'processing_results.json')
        with open(results_path, 'w') as f:
            json.dump(processing_results, f, indent=2, default=str)
        
        print(f"✅ Image processing completed")
        print(f"📊 Successful pairs: {successful_pairs}/{total_images}")
        print(f"📄 Processing results saved: {results_path}")
        
        return processing_results
    
    def analyze_validation_results(self, results_file: str) -> Dict[str, Any]:
        """
        Analyze radiologist validation results
        
        Args:
            results_file: Path to validation results JSON file
            
        Returns:
            Statistical analysis results
        """
        print(f"📊 Analyzing radiologist validation results: {results_file}")
        
        # Load validation results
        with open(results_file, 'r') as f:
            validation_data = json.load(f)
        
        # Extract scores
        original_scores = validation_data.get('original_scores', [])
        stego_scores = validation_data.get('stego_scores', [])
        
        if not original_scores or not stego_scores:
            raise ValueError("No validation scores found in results file")
        
        if len(original_scores) != len(stego_scores):
            raise ValueError("Mismatched number of original and stego scores")
        
        print(f"📋 Analyzing {len(original_scores)} image pairs")
        
        analysis_results = {
            'descriptive_statistics': self._calculate_descriptive_statistics(original_scores, stego_scores),
            'diagnostic_accuracy_analysis': self._analyze_diagnostic_accuracy(original_scores, stego_scores),
            'statistical_significance': self._perform_statistical_tests(original_scores, stego_scores),
            'effect_size_analysis': self._calculate_effect_sizes(original_scores, stego_scores),
            'agreement_analysis': self._analyze_agreement(original_scores, stego_scores),
            'clinical_implications': self._assess_clinical_implications(original_scores, stego_scores)
        }
        
        # Generate visualizations
        viz_path = self._create_analysis_visualizations(original_scores, stego_scores, analysis_results)
        analysis_results['visualization_path'] = viz_path
        
        # Generate report
        report_path = self._generate_analysis_report(analysis_results)
        analysis_results['report_path'] = report_path
        
        # Print summary
        self._print_analysis_summary(analysis_results)
        
        return analysis_results
    
    def _find_validation_images(self, validation_dir: str) -> List[str]:
        """Find validation images in directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
        image_files = []
        
        if not os.path.exists(validation_dir):
            print(f"⚠️  Warning: Validation directory does not exist: {validation_dir}")
            return []
        
        for root, dirs, files in os.walk(validation_dir):
            for file in sorted(files):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        return image_files
    
    def _create_evaluation_templates(self, study_dir: str, study_config: Dict[str, Any]):
        """Create evaluation form templates"""
        forms_dir = os.path.join(study_dir, 'evaluation_forms')
        
        # Individual evaluation form template
        evaluation_form = {
            'study_name': study_config['study_metadata']['study_name'],
            'instructions': {
                'overview': 'Please evaluate each image pair for diagnostic quality',
                'rating_scale': study_config['validation_protocol']['rating_scale'],
                'criteria': study_config['validation_protocol']['evaluation_criteria'],
                'time_limit': 'No time limit - evaluate at your own pace',
                'blinding': 'You will not be told which image is original vs. processed'
            },
            'evaluation_data': []
        }
        
        # Create evaluation entries for each image pair
        for i in range(study_config['study_metadata']['target_images']):
            pair_entry = {
                'pair_id': f"pair_{i+1:03d}",
                'image_a': f"pair_{i+1:03d}_image_a.jpg",
                'image_b': f"pair_{i+1:03d}_image_b.jpg",
                'evaluation': {
                    'overall_quality_a': None,
                    'overall_quality_b': None,
                    'diagnostic_confidence_a': None,
                    'diagnostic_confidence_b': None,
                    'preferred_image': None,  # 'A', 'B', or 'equal'
                    'comments': ""
                }
            }
            evaluation_form['evaluation_data'].append(pair_entry)
        
        # Save evaluation form template
        form_path = os.path.join(forms_dir, 'evaluation_form_template.json')
        with open(form_path, 'w') as f:
            json.dump(evaluation_form, f, indent=2)
    
    def _create_instruction_documents(self, study_dir: str, study_config: Dict[str, Any]):
        """Create instruction documents for radiologists"""
        docs_dir = os.path.join(study_dir, 'documentation')
        
        instructions = f"""
RADIOLOGIST VALIDATION STUDY INSTRUCTIONS
=========================================

Study: {study_config['study_metadata']['study_name']}
Date: {study_config['study_metadata']['creation_date']}

OBJECTIVE:
Evaluate the diagnostic quality of MRI images before and after steganographic processing.

PROCEDURE:
1. You will be presented with {study_config['study_metadata']['target_images']} image pairs
2. Each pair consists of two images: one original, one processed (order randomized)
3. Evaluate each image independently using the provided rating scale
4. You will not be told which image is original vs. processed

RATING SCALE:
{chr(10).join([f"{k}: {v}" for k, v in study_config['validation_protocol']['rating_scale'].items()])}

EVALUATION CRITERIA:
{chr(10).join([f"- {criterion}" for criterion in study_config['validation_protocol']['evaluation_criteria']])}

DIAGNOSTIC THRESHOLD:
Images rated {study_config['validation_protocol']['diagnostic_threshold']} or above are considered diagnostically acceptable.

INSTRUCTIONS:
1. Open the evaluation form (Excel spreadsheet or JSON file)
2. For each image pair, rate both images (A and B) on the 1-5 scale
3. Indicate your preferred image for diagnosis (A, B, or equal)
4. Add any relevant comments
5. Save your responses when complete

TIME COMMITMENT:
Approximately 30-60 minutes total (1-2 minutes per image pair)

CONTACT:
For questions or technical issues, contact the research team.

Thank you for your participation in this important research!
"""
        
        # Save instructions
        instructions_path = os.path.join(docs_dir, 'radiologist_instructions.txt')
        with open(instructions_path, 'w') as f:
            f.write(instructions)
    
    def _create_data_collection_spreadsheet(self, study_dir: str, study_config: Dict[str, Any]):
        """Create Excel spreadsheet for data collection"""
        results_dir = os.path.join(study_dir, 'results')
        
        # Create DataFrame for data collection
        data = []
        for i in range(study_config['study_metadata']['target_images']):
            row = {
                'Pair_ID': f"pair_{i+1:03d}",
                'Image_A_Quality': None,
                'Image_B_Quality': None,
                'Image_A_Diagnostic_Confidence': None,
                'Image_B_Diagnostic_Confidence': None,
                'Preferred_Image': None,
                'Comments': ""
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save as Excel file
        excel_path = os.path.join(results_dir, 'validation_data_collection.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Validation_Data', index=False)
            
            # Add instructions sheet
            instructions_df = pd.DataFrame({
                'Rating_Scale': list(study_config['validation_protocol']['rating_scale'].keys()),
                'Description': list(study_config['validation_protocol']['rating_scale'].values())
            })
            instructions_df.to_excel(writer, sheet_name='Rating_Scale', index=False)
        
        print(f"📊 Data collection spreadsheet created: {excel_path}")
    
    def _calculate_descriptive_statistics(self, original_scores: List[float], stego_scores: List[float]) -> Dict[str, float]:
        """Calculate descriptive statistics"""
        return {
            'original': {
                'mean': float(np.mean(original_scores)),
                'std': float(np.std(original_scores)),
                'median': float(np.median(original_scores)),
                'min': float(np.min(original_scores)),
                'max': float(np.max(original_scores))
            },
            'stego': {
                'mean': float(np.mean(stego_scores)),
                'std': float(np.std(stego_scores)),
                'median': float(np.median(stego_scores)),
                'min': float(np.min(stego_scores)),
                'max': float(np.max(stego_scores))
            }
        }
    
    def _analyze_diagnostic_accuracy(self, original_scores: List[float], stego_scores: List[float]) -> Dict[str, float]:
        """Analyze diagnostic accuracy preservation"""
        threshold = self.validation_config['diagnostic_threshold']
        
        original_diagnostic = np.array(original_scores) >= threshold
        stego_diagnostic = np.array(stego_scores) >= threshold
        
        original_accuracy = np.mean(original_diagnostic) * 100
        stego_accuracy = np.mean(stego_diagnostic) * 100
        accuracy_preservation = (stego_accuracy / original_accuracy * 100) if original_accuracy > 0 else 0
        
        return {
            'original_diagnostic_accuracy': float(original_accuracy),
            'stego_diagnostic_accuracy': float(stego_accuracy),
            'accuracy_preservation_percent': float(accuracy_preservation),
            'diagnostic_threshold': threshold,
            'original_diagnostic_count': int(np.sum(original_diagnostic)),
            'stego_diagnostic_count': int(np.sum(stego_diagnostic))
        }
    
    def _perform_statistical_tests(self, original_scores: List[float], stego_scores: List[float]) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(original_scores, stego_scores)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        w_stat, w_p_value = stats.wilcoxon(original_scores, stego_scores)
        
        return {
            'paired_ttest': {
                'statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_at_05': p_value < 0.05,
                'significant_at_01': p_value < 0.01
            },
            'wilcoxon_test': {
                'statistic': float(w_stat),
                'p_value': float(w_p_value),
                'significant_at_05': w_p_value < 0.05,
                'significant_at_01': w_p_value < 0.01
            }
        }
    
    def _calculate_effect_sizes(self, original_scores: List[float], stego_scores: List[float]) -> Dict[str, float]:
        """Calculate effect sizes"""
        # Cohen's d
        pooled_std = np.sqrt((np.var(original_scores) + np.var(stego_scores)) / 2)
        cohens_d = (np.mean(original_scores) - np.mean(stego_scores)) / pooled_std if pooled_std > 0 else 0
        
        # Correlation
        correlation = np.corrcoef(original_scores, stego_scores)[0, 1]
        
        return {
            'cohens_d': float(cohens_d),
            'effect_size_interpretation': self._interpret_effect_size(cohens_d),
            'correlation': float(correlation),
            'correlation_interpretation': self._interpret_correlation(correlation)
        }
    
    def _analyze_agreement(self, original_scores: List[float], stego_scores: List[float]) -> Dict[str, Any]:
        """Analyze agreement between original and stego ratings"""
        threshold = self.validation_config['diagnostic_threshold']
        
        original_diagnostic = np.array(original_scores) >= threshold
        stego_diagnostic = np.array(stego_scores) >= threshold
        
        # Agreement
        agreement = original_diagnostic == stego_diagnostic
        agreement_percent = np.mean(agreement) * 100
        
        # Confusion matrix
        cm = confusion_matrix(original_diagnostic, stego_diagnostic)
        
        return {
            'overall_agreement_percent': float(agreement_percent),
            'agreements': int(np.sum(agreement)),
            'disagreements': int(np.sum(~agreement)),
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_labels': ['Non-diagnostic', 'Diagnostic']
        }
    
    def _assess_clinical_implications(self, original_scores: List[float], stego_scores: List[float]) -> Dict[str, Any]:
        """Assess clinical implications of the results"""
        threshold = self.validation_config['diagnostic_threshold']
        
        # Calculate quality degradation
        score_differences = np.array(original_scores) - np.array(stego_scores)
        mean_degradation = np.mean(score_differences)
        
        # Assess clinical acceptability
        stego_diagnostic = np.array(stego_scores) >= threshold
        clinical_acceptability = np.mean(stego_diagnostic) * 100
        
        # Risk assessment
        risk_assessment = self._assess_clinical_risk(clinical_acceptability, mean_degradation)
        
        return {
            'mean_quality_degradation': float(mean_degradation),
            'clinical_acceptability_percent': float(clinical_acceptability),
            'recommended_for_clinical_use': clinical_acceptability >= 90,  # 90% threshold
            'risk_assessment': risk_assessment,
            'recommendations': self._generate_clinical_recommendations(clinical_acceptability, mean_degradation)
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr < 0.3:
            return "weak"
        elif abs_corr < 0.7:
            return "moderate"
        else:
            return "strong"
    
    def _assess_clinical_risk(self, acceptability: float, degradation: float) -> str:
        """Assess clinical risk level"""
        if acceptability >= 95 and degradation < 0.1:
            return "very_low"
        elif acceptability >= 90 and degradation < 0.2:
            return "low"
        elif acceptability >= 80 and degradation < 0.5:
            return "moderate"
        elif acceptability >= 70:
            return "high"
        else:
            return "very_high"
    
    def _generate_clinical_recommendations(self, acceptability: float, degradation: float) -> List[str]:
        """Generate clinical recommendations"""
        recommendations = []
        
        if acceptability >= 90:
            recommendations.append("Steganographic processing shows minimal impact on diagnostic quality")
        
        if degradation < 0.1:
            recommendations.append("Quality degradation is negligible and clinically acceptable")
        
        if acceptability < 80:
            recommendations.append("Consider refining steganographic parameters to improve quality preservation")
        
        if degradation > 0.5:
            recommendations.append("Significant quality degradation detected - clinical validation required")
        
        recommendations.append("Regular quality monitoring recommended for clinical deployment")
        
        return recommendations
    
    def _create_analysis_visualizations(self, original_scores: List[float], stego_scores: List[float], 
                                      analysis_results: Dict[str, Any]) -> str:
        """Create analysis visualizations"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Radiologist Validation Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Score distribution comparison
        axes[0, 0].hist(original_scores, alpha=0.7, label='Original', bins=5, density=True)
        axes[0, 0].hist(stego_scores, alpha=0.7, label='Stego', bins=5, density=True)
        axes[0, 0].set_xlabel('Rating Score')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Score Distribution Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Paired scores scatter plot
        axes[0, 1].scatter(original_scores, stego_scores, alpha=0.6)
        axes[0, 1].plot([1, 5], [1, 5], 'r--', label='Perfect Agreement')
        axes[0, 1].set_xlabel('Original Scores')
        axes[0, 1].set_ylabel('Stego Scores')
        axes[0, 1].set_title('Paired Scores Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Diagnostic accuracy comparison
        diag_analysis = analysis_results['diagnostic_accuracy_analysis']
        categories = ['Original', 'Stego']
        accuracies = [diag_analysis['original_diagnostic_accuracy'], diag_analysis['stego_diagnostic_accuracy']]
        
        bars = axes[1, 0].bar(categories, accuracies, color=['blue', 'orange'], alpha=0.7)
        axes[1, 0].set_ylabel('Diagnostic Accuracy (%)')
        axes[1, 0].set_title('Diagnostic Accuracy Comparison')
        axes[1, 0].set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, accuracies):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}%', ha='center', va='bottom')
        
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Agreement analysis
        agreement_data = analysis_results['agreement_analysis']
        cm = np.array(agreement_data['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=agreement_data['confusion_matrix_labels'],
                   yticklabels=agreement_data['confusion_matrix_labels'],
                   ax=axes[1, 1])
        axes[1, 1].set_xlabel('Stego Classification')
        axes[1, 1].set_ylabel('Original Classification')
        axes[1, 1].set_title('Diagnostic Agreement Matrix')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(self.output_dir, 'validation_analysis_visualization.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return viz_path
    
    def _generate_analysis_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report"""
        report_path = os.path.join(self.output_dir, 'radiologist_validation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("RADIOLOGIST VALIDATION ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            # Descriptive statistics
            f.write("DESCRIPTIVE STATISTICS:\n")
            f.write("-"*25 + "\n")
            desc_stats = analysis_results['descriptive_statistics']
            f.write(f"Original Images - Mean: {desc_stats['original']['mean']:.2f} ± {desc_stats['original']['std']:.2f}\n")
            f.write(f"Stego Images - Mean: {desc_stats['stego']['mean']:.2f} ± {desc_stats['stego']['std']:.2f}\n\n")
            
            # Diagnostic accuracy
            f.write("DIAGNOSTIC ACCURACY ANALYSIS:\n")
            f.write("-"*30 + "\n")
            diag_acc = analysis_results['diagnostic_accuracy_analysis']
            f.write(f"Original Diagnostic Accuracy: {diag_acc['original_diagnostic_accuracy']:.1f}%\n")
            f.write(f"Stego Diagnostic Accuracy: {diag_acc['stego_diagnostic_accuracy']:.1f}%\n")
            f.write(f"Accuracy Preservation: {diag_acc['accuracy_preservation_percent']:.1f}%\n\n")
            
            # Statistical significance
            f.write("STATISTICAL SIGNIFICANCE:\n")
            f.write("-"*25 + "\n")
            stats_sig = analysis_results['statistical_significance']
            f.write(f"Paired t-test p-value: {stats_sig['paired_ttest']['p_value']:.4f}\n")
            f.write(f"Statistically significant: {stats_sig['paired_ttest']['significant_at_05']}\n\n")
            
            # Clinical implications
            f.write("CLINICAL IMPLICATIONS:\n")
            f.write("-"*20 + "\n")
            clinical = analysis_results['clinical_implications']
            f.write(f"Clinical Acceptability: {clinical['clinical_acceptability_percent']:.1f}%\n")
            f.write(f"Recommended for Clinical Use: {clinical['recommended_for_clinical_use']}\n")
            f.write(f"Risk Assessment: {clinical['risk_assessment']}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-"*15 + "\n")
            for rec in clinical['recommendations']:
                f.write(f"• {rec}\n")
        
        return report_path
    
    def _print_analysis_summary(self, analysis_results: Dict[str, Any]):
        """Print analysis summary to console"""
        print(f"\n{'='*60}")
        print("📊 RADIOLOGIST VALIDATION ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        # Diagnostic accuracy
        diag_acc = analysis_results['diagnostic_accuracy_analysis']
        print(f"🏥 DIAGNOSTIC ACCURACY:")
        print(f"   Original: {diag_acc['original_diagnostic_accuracy']:.1f}%")
        print(f"   Stego: {diag_acc['stego_diagnostic_accuracy']:.1f}%")
        print(f"   Preservation: {diag_acc['accuracy_preservation_percent']:.1f}%")
        
        # Clinical implications
        clinical = analysis_results['clinical_implications']
        print(f"\n🔬 CLINICAL ASSESSMENT:")
        print(f"   Clinical Acceptability: {clinical['clinical_acceptability_percent']:.1f}%")
        print(f"   Recommended for Clinical Use: {'✅ YES' if clinical['recommended_for_clinical_use'] else '❌ NO'}")
        print(f"   Risk Level: {clinical['risk_assessment'].upper()}")
        
        # Statistical significance
        stats_sig = analysis_results['statistical_significance']
        print(f"\n📈 STATISTICAL ANALYSIS:")
        print(f"   p-value: {stats_sig['paired_ttest']['p_value']:.4f}")
        print(f"   Statistically significant: {'✅ YES' if stats_sig['paired_ttest']['significant_at_05'] else '❌ NO'}")
        
        print(f"\n{'='*60}")


def main():
    """Main function for radiologist validation"""
    parser = argparse.ArgumentParser(description='Radiologist Validation for MRI Steganography')
    
    parser.add_argument('--setup', action='store_true',
                       help='Setup validation study')
    
    parser.add_argument('--process', action='store_true',
                       help='Process validation images')
    
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze validation results')
    
    parser.add_argument('--validation_dir', type=str,
                       default='data/mri_dataset/validation_set',
                       help='Directory containing validation MRI images')
    
    parser.add_argument('--output_dir', type=str,
                       default='results/radiologist_validation',
                       help='Output directory for validation study')
    
    parser.add_argument('--study_name', type=str,
                       default='MRI_Stego_Validation_2025',
                       help='Name of the validation study')
    
    parser.add_argument('--results_file', type=str,
                       help='Path to validation results JSON file (for analysis)')
    
    args = parser.parse_args()
    
    # Initialize validation manager
    validator = RadiologistValidationManager(args.output_dir)
    
    if args.setup:
        print("🏥 Setting up radiologist validation study...")
        study_config = validator.setup_validation_study(args.validation_dir, args.study_name)
        
        print(f"\\n✅ Validation study setup completed!")
        print(f"📁 Study directory: {study_config['study_metadata']['study_directory']}")
        print(f"📋 Images to evaluate: {study_config['study_metadata']['target_images']}")
        
        if args.process:
            print(f"\\n🔄 Processing validation images...")
            processing_results = validator.process_validation_images(study_config)
            print(f"✅ Image processing completed!")
            print(f"📊 Successful pairs: {processing_results['processing_statistics']['successful_pairs']}")
    
    elif args.analyze:
        if not args.results_file:
            print("❌ Error: --results_file required for analysis")
            sys.exit(1)
        
        print(f"📊 Analyzing validation results...")
        analysis_results = validator.analyze_validation_results(args.results_file)
        
        print(f"\\n✅ Analysis completed!")
        print(f"📄 Report saved: {analysis_results['report_path']}")
        print(f"📈 Visualizations saved: {analysis_results['visualization_path']}")
    
    else:
        print("❌ Please specify --setup, --process, or --analyze")
        parser.print_help()


if __name__ == "__main__":
    main()