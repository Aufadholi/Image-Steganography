"""
Adaptive Multi-Objective Threshold Optimization for Hybrid Steganography
Enhanced optimization layer for existing hybrid edge detection + LBP + reversible LSB system
Specialized for MRI image characteristics and medical imaging requirements
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional
import random
from datetime import datetime
from scipy import ndimage
from skimage import filters


class MRIImageCharacteristicsAnalyzer:
    """Analyze MRI image characteristics to guide parameter optimization"""
    
    def __init__(self):
        self.feature_weights = {
            'texture_density': 0.25,
            'edge_distribution': 0.2,
            'noise_level': 0.15,
            'local_variance': 0.15,
            'gradient_magnitude': 0.1,
            'contrast_variation': 0.1,
            'anatomical_complexity': 0.05
        }
        
        # MRI-specific thresholds and parameters
        self.mri_params = {
            'brain_tissue_range': (50, 200),  # Typical brain tissue intensity range
            'noise_threshold': 10,  # MRI noise level threshold
            'edge_sensitivity': 0.8,  # Edge detection sensitivity for MRI
            'texture_complexity_threshold': 0.5
        }
    
    def analyze(self, image: np.ndarray) -> Dict:
        """
        Comprehensive MRI image analysis for adaptive optimization
        
        Args:
            image: Input MRI cover image
            
        Returns:
            Dictionary of MRI-specific image characteristics
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate MRI-specific image characteristics
        characteristics = {
            'texture_density': self._calculate_texture_density(gray),
            'edge_distribution': self._analyze_edge_distribution(gray),
            'noise_level': self._estimate_mri_noise_level(gray),
            'local_variance': self._calculate_local_variance(gray),
            'gradient_magnitude': self._calculate_gradient_magnitude(gray),
            'contrast_variation': self._calculate_contrast_variation(gray),
            'anatomical_complexity': self._analyze_anatomical_complexity(gray),
            'brain_tissue_ratio': self._calculate_brain_tissue_ratio(gray),
            'intensity_distribution': self._analyze_intensity_distribution(gray),
            'image_complexity': 0.0,  # Will be calculated
            'mri_specific_hints': {}  # MRI-specific recommendations
        }
        
        # Calculate composite complexity score for MRI
        characteristics['image_complexity'] = self._calculate_mri_complexity_score(characteristics)
        
        # Generate MRI-specific threshold recommendations
        characteristics['mri_specific_hints'] = self._generate_mri_threshold_hints(characteristics)
        
        return characteristics
    
    def _calculate_texture_density(self, gray: np.ndarray) -> float:
        """Calculate texture density using local binary patterns"""
        # Simple LBP-based texture measurement
        height, width = gray.shape
        texture_sum = 0.0
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = gray[i, j]
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                binary_pattern = sum([1 if n >= center else 0 for n in neighbors])
                texture_sum += binary_pattern / 8.0
        
        return texture_sum / ((height-2) * (width-2))
    
    def _analyze_edge_distribution(self, gray: np.ndarray) -> float:
        """Analyze edge distribution using Canny edge detection"""
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        return edge_density
    
    def _estimate_noise_level(self, gray: np.ndarray) -> float:
        """Estimate noise level using Laplacian variance"""
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-1 range
        return min(laplacian_var / 1000.0, 1.0)
    
    def _calculate_local_variance(self, gray: np.ndarray) -> float:
        """Calculate local variance across image regions"""
        # Divide image into blocks and calculate variance
        block_size = 32
        height, width = gray.shape
        variances = []
        
        for i in range(0, height - block_size, block_size):
            for j in range(0, width - block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                variances.append(np.var(block))
        
        return np.mean(variances) / 255.0  # Normalize
    
    def _calculate_gradient_magnitude(self, gray: np.ndarray) -> float:
        """Calculate average gradient magnitude"""
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(gradient_magnitude) / 255.0  # Normalize
    
    def _estimate_mri_noise_level(self, gray: np.ndarray) -> float:
        """Estimate MRI-specific noise level"""
        # Calculate noise in background regions (low intensity areas)
        background_mask = gray < np.percentile(gray, 10)
        if np.sum(background_mask) > 0:
            background_std = np.std(gray[background_mask])
            noise_level = background_std / 255.0
        else:
            # Fallback to Laplacian variance method
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            noise_level = min(laplacian_var / 1000.0, 1.0)
        
        return noise_level
    
    def _calculate_contrast_variation(self, gray: np.ndarray) -> float:
        """Calculate contrast variation across the MRI image"""
        # Use local standard deviation as contrast measure
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        # Local mean
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # Local standard deviation (contrast)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        local_std = np.sqrt(local_variance)
        
        # Return coefficient of variation of local contrasts
        contrast_variation = np.std(local_std) / (np.mean(local_std) + 1e-8)
        return min(contrast_variation, 1.0)
    
    def _analyze_anatomical_complexity(self, gray: np.ndarray) -> float:
        """Analyze anatomical complexity specific to MRI"""
        # Use multi-scale edge detection to assess anatomical complexity
        scales = [1, 2, 3]
        complexity_scores = []
        
        for scale in scales:
            # Apply Gaussian blur at different scales
            blurred = cv2.GaussianBlur(gray, (scale*2+1, scale*2+1), scale)
            
            # Detect edges at this scale
            edges = cv2.Canny(blurred, 30, 100)
            edge_density = np.sum(edges > 0) / edges.size
            complexity_scores.append(edge_density)
        
        # Combine scores from different scales
        anatomical_complexity = np.mean(complexity_scores)
        return anatomical_complexity
    
    def _calculate_brain_tissue_ratio(self, gray: np.ndarray) -> float:
        """Calculate ratio of brain tissue vs background"""
        brain_min, brain_max = self.mri_params['brain_tissue_range']
        brain_tissue_mask = (gray >= brain_min) & (gray <= brain_max)
        brain_ratio = np.sum(brain_tissue_mask) / gray.size
        return brain_ratio
    
    def _analyze_intensity_distribution(self, gray: np.ndarray) -> Dict:
        """Analyze intensity distribution characteristics"""
        # Calculate histogram statistics
        hist, _ = np.histogram(gray, bins=256, range=(0, 255))
        hist_normalized = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-8))
        
        # Calculate skewness and kurtosis approximations
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Mode (most frequent intensity)
        mode_intensity = np.argmax(hist)
        
        return {
            'entropy': entropy / 8.0,  # Normalized
            'mean': mean_intensity / 255.0,
            'std': std_intensity / 255.0,
            'mode': mode_intensity / 255.0,
            'dynamic_range': (np.max(gray) - np.min(gray)) / 255.0
        }
    
    def _calculate_mri_complexity_score(self, characteristics: Dict) -> float:
        """Calculate MRI-specific composite complexity score"""
        score = 0.0
        for feature, weight in self.feature_weights.items():
            if feature in characteristics:
                score += characteristics[feature] * weight
        return score
    
    def _generate_mri_threshold_hints(self, characteristics: Dict) -> Dict:
        """Generate MRI-specific threshold recommendations"""
        complexity = characteristics['image_complexity']
        brain_ratio = characteristics['brain_tissue_ratio']
        noise_level = characteristics['noise_level']
        anatomical_complexity = characteristics['anatomical_complexity']
        
        # Adaptive threshold recommendations for MRI
        if complexity > 0.7 or anatomical_complexity > 0.3:  # High complexity MRI
            hints = {
                'edge_threshold_range': [0.3, 0.5],  # More conservative for complex anatomy
                'texture_threshold_range': [0.4, 0.6],
                'capacity_ratio_range': [0.05, 0.10],  # Lower capacity for safety
                'preprocessing_recommendation': 'aggressive_denoising',
                'roi_safety_margin': 7  # Larger safety margin
            }
        elif complexity > 0.4 or brain_ratio > 0.6:  # Medium complexity MRI
            hints = {
                'edge_threshold_range': [0.2, 0.4],
                'texture_threshold_range': [0.3, 0.5],
                'capacity_ratio_range': [0.08, 0.15],
                'preprocessing_recommendation': 'moderate_denoising',
                'roi_safety_margin': 5
            }
        else:  # Low complexity MRI (unusual but possible)
            hints = {
                'edge_threshold_range': [0.1, 0.3],
                'texture_threshold_range': [0.2, 0.4],
                'capacity_ratio_range': [0.10, 0.20],
                'preprocessing_recommendation': 'light_denoising',
                'roi_safety_margin': 3
            }
        
        # Adjust based on noise level
        if noise_level > 0.3:
            hints['preprocessing_recommendation'] = 'aggressive_denoising'
            hints['edge_threshold_range'] = [
                hints['edge_threshold_range'][0] + 0.1,
                hints['edge_threshold_range'][1] + 0.1
            ]
        
        # Add MRI-specific recommendations
        hints['mri_specific'] = {
            'avoid_ventricles': True,
            'prioritize_gray_matter': brain_ratio > 0.5,
            'use_anatomical_segmentation': anatomical_complexity > 0.2,
            'bias_field_correction': True,
            'intensity_normalization': True
        }
        
        return hints
    
    def _generate_threshold_hints(self, characteristics: Dict) -> Dict:
        """Generate threshold recommendations based on image characteristics"""
        complexity = characteristics['image_complexity']
        texture_density = characteristics['texture_density']
        edge_distribution = characteristics['edge_distribution']
        
        # Adaptive threshold recommendations
        if complexity > 0.7:  # High complexity
            hints = {
                'edge_threshold_range': [0.4, 0.7],
                'texture_threshold_range': [0.5, 0.8],
                'capacity_ratio_range': [0.08, 0.15]
            }
        elif complexity > 0.4:  # Medium complexity
            hints = {
                'edge_threshold_range': [0.3, 0.6],
                'texture_threshold_range': [0.4, 0.7],
                'capacity_ratio_range': [0.06, 0.12]
            }
        else:  # Low complexity
            hints = {
                'edge_threshold_range': [0.2, 0.5],
                'texture_threshold_range': [0.3, 0.6],
                'capacity_ratio_range': [0.04, 0.10]
            }
        
        return hints


class MultiObjectiveOptimizer:
    """Multi-objective optimization for steganography parameters"""
    
    def __init__(self, population_size: int = 20, max_iterations: int = 50):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.objectives = ['psnr', 'capacity', 'imperceptibility']
        
    def optimize(self, image_characteristics: Dict, payload_size: int, 
                base_config: Dict) -> Dict:
        """
        Multi-objective PSO optimization for threshold parameters
        
        Args:
            image_characteristics: Image analysis results
            payload_size: Size of payload to embed
            base_config: Current configuration as baseline
            
        Returns:
            Optimized configuration parameters
        """
        print("🔄 Starting adaptive multi-objective optimization...")
        
        # Get search space from image characteristics
        search_space = self._define_search_space(image_characteristics, base_config)
        
        # Initialize particle swarm
        particles = self._initialize_particles(search_space)
        
        # PSO optimization
        best_solutions = []
        for iteration in range(self.max_iterations):
            # Evaluate fitness for each particle
            fitness_scores = []
            for particle in particles:
                fitness = self._evaluate_fitness(particle, image_characteristics, payload_size)
                fitness_scores.append(fitness)
            
            # Update best solutions (Pareto front)
            best_solutions.extend(self._update_pareto_front(particles, fitness_scores))
            
            # Update particle velocities and positions
            particles = self._update_particles(particles, best_solutions)
            
            # Progress feedback
            if iteration % 10 == 0:
                best_fitness = max(fitness_scores, key=lambda x: x['composite_score'])
                print(f"   Iteration {iteration}: Best composite score = {best_fitness['composite_score']:.4f}")
        
        # Select optimal solution from Pareto front
        optimal_config = self._select_optimal_solution(best_solutions, base_config)
        
        print("✅ Optimization completed!")
        return optimal_config
    
    def _define_search_space(self, characteristics: Dict, base_config: Dict) -> Dict:
        """Define parameter search space based on image characteristics"""
        hints = characteristics.get('optimal_threshold_hint', {})
        
        return {
            'edge_threshold': hints.get('edge_threshold_range', [0.2, 0.6]),
            'texture_threshold': hints.get('texture_threshold_range', [0.3, 0.7]),
            'max_capacity_ratio': hints.get('capacity_ratio_range', [0.05, 0.15])
        }
    
    def _initialize_particles(self, search_space: Dict) -> List[Dict]:
        """Initialize particle swarm with random positions"""
        particles = []
        for _ in range(self.population_size):
            particle = {}
            for param, (min_val, max_val) in search_space.items():
                particle[param] = random.uniform(min_val, max_val)
            particles.append(particle)
        return particles
    
    def _evaluate_fitness(self, particle: Dict, characteristics: Dict, 
                         payload_size: int) -> Dict:
        """Evaluate fitness of parameter configuration"""
        # Simulate embedding performance (simplified for optimization)
        complexity = characteristics['image_complexity']
        
        # Objective 1: PSNR estimation
        edge_quality = 1.0 - abs(particle['edge_threshold'] - 0.4)  # Optimal around 0.4
        texture_quality = 1.0 - abs(particle['texture_threshold'] - 0.5)  # Optimal around 0.5
        psnr_score = (edge_quality + texture_quality) / 2.0
        
        # Objective 2: Capacity utilization
        capacity_score = min(particle['max_capacity_ratio'] * 10, 1.0)  # Prefer higher capacity
        
        # Objective 3: Imperceptibility (lower detection risk)
        imperceptibility_score = 1.0 - (particle['max_capacity_ratio'] * 0.5)  # Trade-off with capacity
        
        # Composite score (weighted combination)
        weights = [0.4, 0.3, 0.3]  # PSNR, Capacity, Imperceptibility
        composite_score = (
            weights[0] * psnr_score +
            weights[1] * capacity_score +
            weights[2] * imperceptibility_score
        )
        
        return {
            'psnr_score': psnr_score,
            'capacity_score': capacity_score,
            'imperceptibility_score': imperceptibility_score,
            'composite_score': composite_score,
            'parameters': particle.copy()
        }
    
    def _update_pareto_front(self, particles: List[Dict], 
                           fitness_scores: List[Dict]) -> List[Dict]:
        """Update Pareto front with non-dominated solutions"""
        pareto_solutions = []
        
        for i, fitness in enumerate(fitness_scores):
            is_dominated = False
            for j, other_fitness in enumerate(fitness_scores):
                if i != j:
                    # Check if solution i is dominated by solution j
                    if (other_fitness['psnr_score'] >= fitness['psnr_score'] and
                        other_fitness['capacity_score'] >= fitness['capacity_score'] and
                        other_fitness['imperceptibility_score'] >= fitness['imperceptibility_score'] and
                        (other_fitness['psnr_score'] > fitness['psnr_score'] or
                         other_fitness['capacity_score'] > fitness['capacity_score'] or
                         other_fitness['imperceptibility_score'] > fitness['imperceptibility_score'])):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_solutions.append(fitness)
        
        return pareto_solutions
    
    def _update_particles(self, particles: List[Dict], 
                         best_solutions: List[Dict]) -> List[Dict]:
        """Update particle positions using PSO dynamics"""
        # Simplified PSO update (for demonstration)
        updated_particles = []
        
        for particle in particles:
            new_particle = {}
            for param in particle:
                # Add small random perturbation
                perturbation = random.uniform(-0.05, 0.05)
                new_value = particle[param] + perturbation
                
                # Keep within bounds (simplified)
                new_particle[param] = max(0.1, min(0.9, new_value))
            
            updated_particles.append(new_particle)
        
        return updated_particles
    
    def _select_optimal_solution(self, pareto_solutions: List[Dict], 
                               base_config: Dict) -> Dict:
        """Select optimal solution from Pareto front"""
        if not pareto_solutions:
            return base_config
        
        # Select solution with highest composite score
        best_solution = max(pareto_solutions, key=lambda x: x['composite_score'])
        
        # Merge with base configuration
        optimized_config = base_config.copy()
        optimized_config.update(best_solution['parameters'])
        
        # Add optimization metadata
        optimized_config['optimization_info'] = {
            'method': 'multi_objective_pso',
            'composite_score': best_solution['composite_score'],
            'psnr_score': best_solution['psnr_score'],
            'capacity_score': best_solution['capacity_score'],
            'imperceptibility_score': best_solution['imperceptibility_score'],
            'timestamp': datetime.now().isoformat()
        }
        
        return optimized_config


class AdaptiveHybridSteganography:
    """Main class for adaptive hybrid steganography"""
    
    def __init__(self):
        self.analyzer = MRIImageCharacteristicsAnalyzer()
        self.optimizer = MultiObjectiveOptimizer()
        
    def optimize_parameters(self, cover_image: np.ndarray, payload_size: int, 
                          base_config: Dict) -> Dict:
        """
        Main optimization function - enhance existing configuration
        
        Args:
            cover_image: Cover image for embedding
            payload_size: Size of payload in bytes
            base_config: Current configuration
            
        Returns:
            Optimized configuration
        """
        print("🧠 Analyzing image characteristics...")
        
        # Analyze image characteristics
        characteristics = self.analyzer.analyze(cover_image)
        
        print(f"   Image complexity score: {characteristics['image_complexity']:.3f}")
        print(f"   Texture density: {characteristics['texture_density']:.3f}")
        print(f"   Edge distribution: {characteristics['edge_distribution']:.3f}")
        
        # Optimize parameters
        optimized_config = self.optimizer.optimize(
            characteristics, payload_size, base_config
        )
        
        print("📊 Optimization results:")
        opt_info = optimized_config.get('optimization_info', {})
        print(f"   Composite score: {opt_info.get('composite_score', 0):.4f}")
        print(f"   PSNR score: {opt_info.get('psnr_score', 0):.4f}")
        print(f"   Capacity score: {opt_info.get('capacity_score', 0):.4f}")
        print(f"   Imperceptibility score: {opt_info.get('imperceptibility_score', 0):.4f}")
        
        return optimized_config