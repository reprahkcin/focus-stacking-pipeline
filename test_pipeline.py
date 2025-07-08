#!/usr/bin/env python3
"""
Comprehensive Testing Framework for Focus-Stacking Pipeline
Tests with real image sets and provides detailed analysis
"""

import cv2
import numpy as np
import yaml
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from focus_stack import FocusStacker
import subprocess
import json


class PipelineTester:
    """Comprehensive testing framework for focus-stacking pipeline"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.stacker = FocusStacker(config_path)
        self._setup_logging()
        self.test_results = {}

    def _load_config(self, config_path: str) -> dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Setup logging for testing"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def analyze_image_set(self, image_paths: List[Path]) -> Dict:
        """Analyze a set of images for focus-stacking suitability"""
        self.logger.info(f"Analyzing {len(image_paths)} images...")

        analysis = {
            'image_count': len(image_paths),
            'file_sizes': [],
            'dimensions': [],
            'focus_ranges': [],
            'sharpness_scores': [],
            'alignment_quality': [],
            'total_size_mb': 0
        }

        images = []
        for i, path in enumerate(image_paths):
            # Load image
            img = cv2.imread(str(path))
            if img is None:
                self.logger.warning(f"Failed to load image: {path}")
                continue

            # Basic info
            file_size = path.stat().st_size / (1024 * 1024)  # MB
            analysis['file_sizes'].append(file_size)
            analysis['total_size_mb'] += file_size
            analysis['dimensions'].append(img.shape[:2])

            # Calculate sharpness
            sharpness = self._calculate_sharpness(img)
            analysis['sharpness_scores'].append(sharpness)

            images.append(img)

        if len(images) < 2:
            self.logger.error("Need at least 2 images for focus-stacking")
            return analysis

        # Analyze focus range
        analysis['focus_ranges'] = self._analyze_focus_range(
            analysis['sharpness_scores'])

        # Test alignment
        analysis['alignment_quality'] = self._test_alignment(images)

        # Overall assessment
        analysis['suitability_score'] = self._calculate_suitability_score(
            analysis)

        return analysis

    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate sharpness score for an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
            image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        return np.var(laplacian)

    def _analyze_focus_range(self, sharpness_scores: List[float]) -> Dict:
        """Analyze the focus range of the image set"""
        if not sharpness_scores:
            return {'min': 0, 'max': 0, 'range': 0, 'coverage': 'poor'}

        min_sharp = min(sharpness_scores)
        max_sharp = max(sharpness_scores)
        sharp_range = max_sharp - min_sharp

        # Assess focus coverage
        if sharp_range < 100:
            coverage = 'poor'
        elif sharp_range < 500:
            coverage = 'fair'
        elif sharp_range < 1000:
            coverage = 'good'
        else:
            coverage = 'excellent'

        return {
            'min': min_sharp,
            'max': max_sharp,
            'range': sharp_range,
            'coverage': coverage
        }

    def _test_alignment(self, images: List[np.ndarray]) -> Dict:
        """Test alignment quality between images"""
        if len(images) < 2:
            return {'quality': 'poor', 'max_shift': 0, 'issues': ['insufficient_images']}

        issues = []
        max_shifts = []

        # Test alignment between consecutive images
        for i in range(1, len(images)):
            try:
                # Simple alignment test
                shift = self._estimate_shift(images[i-1], images[i])
                max_shifts.append(shift)

                if shift > 50:  # pixels
                    issues.append(f'large_shift_{i}: {shift:.1f}px')

            except Exception as e:
                issues.append(f'alignment_failure_{i}: {str(e)}')

        if not max_shifts:
            return {'quality': 'poor', 'max_shift': 0, 'issues': issues}

        max_shift = max(max_shifts)

        # Assess alignment quality
        if max_shift < 5:
            quality = 'excellent'
        elif max_shift < 20:
            quality = 'good'
        elif max_shift < 50:
            quality = 'fair'
        else:
            quality = 'poor'

        return {
            'quality': quality,
            'max_shift': max_shift,
            'issues': issues
        }

    def _estimate_shift(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Estimate shift between two images"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Use phase correlation to estimate shift
        shift, error = cv2.phaseCorrelate(
            gray1.astype(np.float32), gray2.astype(np.float32))
        return np.sqrt(shift[0]**2 + shift[1]**2)

    def _calculate_suitability_score(self, analysis: Dict) -> float:
        """Calculate overall suitability score (0-100)"""
        score = 0

        # Image count (0-20 points)
        count = analysis['image_count']
        if count >= 10:
            score += 20
        elif count >= 5:
            score += 15
        elif count >= 3:
            score += 10
        else:
            score += 5

        # Focus range (0-30 points)
        focus_range = analysis['focus_ranges']['coverage']
        if focus_range == 'excellent':
            score += 30
        elif focus_range == 'good':
            score += 25
        elif focus_range == 'fair':
            score += 15
        else:
            score += 5

        # Alignment quality (0-30 points)
        alignment = analysis['alignment_quality']['quality']
        if alignment == 'excellent':
            score += 30
        elif alignment == 'good':
            score += 25
        elif alignment == 'fair':
            score += 15
        else:
            score += 5

        # Sharpness variation (0-20 points)
        sharpness_scores = analysis['sharpness_scores']
        if len(sharpness_scores) > 1:
            cv = np.std(sharpness_scores) / np.mean(sharpness_scores)
            if cv > 0.3:  # Good variation
                score += 20
            elif cv > 0.1:
                score += 15
            else:
                score += 5

        return min(100, score)

    def test_focus_stacking(self, image_paths: List[Path], output_path: Path) -> Dict:
        """Test focus-stacking on a set of images"""
        self.logger.info(
            f"Testing focus-stacking with {len(image_paths)} images...")

        start_time = time.time()

        # Analyze input
        analysis = self.analyze_image_set(image_paths)

        # Perform focus-stacking
        try:
            success = self.stacker.process_image_set(image_paths, output_path)
            processing_time = time.time() - start_time

            if success:
                # Analyze output
                output_analysis = self._analyze_output(output_path)

                result = {
                    'success': True,
                    'processing_time': processing_time,
                    'input_analysis': analysis,
                    'output_analysis': output_analysis,
                    'performance_metrics': self._calculate_performance_metrics(analysis, processing_time)
                }
            else:
                result = {
                    'success': False,
                    'processing_time': processing_time,
                    'input_analysis': analysis,
                    'error': 'Focus-stacking failed'
                }

        except Exception as e:
            result = {
                'success': False,
                'processing_time': time.time() - start_time,
                'input_analysis': analysis,
                'error': str(e)
            }

        return result

    def _analyze_output(self, output_path: Path) -> Dict:
        """Analyze the output focus-stacked image"""
        img = cv2.imread(str(output_path))
        if img is None:
            return {'error': 'Failed to load output image'}

        # Calculate overall sharpness
        sharpness = self._calculate_sharpness(img)

        # Check for artifacts
        artifacts = self._detect_artifacts(img)

        return {
            'dimensions': img.shape[:2],
            'file_size_mb': output_path.stat().st_size / (1024 * 1024),
            'sharpness': sharpness,
            'artifacts': artifacts
        }

    def _detect_artifacts(self, image: np.ndarray) -> Dict:
        """Detect common focus-stacking artifacts"""
        artifacts = {
            'halos': False,
            'ghosting': False,
            'blending_issues': False
        }

        # Simple artifact detection (can be enhanced)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Check for halos (bright edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        if edge_density > 0.1:  # High edge density might indicate halos
            artifacts['halos'] = True

        return artifacts

    def _calculate_performance_metrics(self, analysis: Dict, processing_time: float) -> Dict:
        """Calculate performance metrics"""
        total_pixels = sum(w * h for w, h in analysis['dimensions'])
        pixels_per_second = total_pixels / processing_time if processing_time > 0 else 0

        return {
            'processing_time': processing_time,
            'total_pixels': total_pixels,
            'pixels_per_second': pixels_per_second,
            'images_per_second': analysis['image_count'] / processing_time if processing_time > 0 else 0
        }

    def generate_test_report(self, test_results: List[Dict], output_dir: Path):
        """Generate comprehensive test report"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create summary
        summary = {
            'total_tests': len(test_results),
            'successful_tests': sum(1 for r in test_results if r.get('success')),
            'average_processing_time': np.mean([r.get('processing_time', 0) for r in test_results]),
            'average_suitability_score': np.mean([r.get('input_analysis', {}).get('suitability_score', 0) for r in test_results])
        }

        # Save detailed results
        with open(output_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2, default=str)

        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Generate visual report
        self._generate_visual_report(test_results, output_dir)

        self.logger.info(f"Test report saved to: {output_dir}")
        return summary

    def _generate_visual_report(self, test_results: List[Dict], output_dir: Path):
        """Generate visual test report"""
        if not test_results:
            return

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Processing time vs image count
        times = [r.get('processing_time', 0) for r in test_results]
        counts = [r.get('input_analysis', {}).get('image_count', 0)
                  for r in test_results]
        axes[0, 0].scatter(counts, times)
        axes[0, 0].set_xlabel('Image Count')
        axes[0, 0].set_ylabel('Processing Time (s)')
        axes[0, 0].set_title('Processing Performance')

        # Suitability scores
        scores = [r.get('input_analysis', {}).get(
            'suitability_score', 0) for r in test_results]
        axes[0, 1].hist(scores, bins=10)
        axes[0, 1].set_xlabel('Suitability Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Suitability Score Distribution')

        # Focus range coverage
        coverages = [r.get('input_analysis', {}).get('focus_ranges', {}).get(
            'coverage', 'poor') for r in test_results]
        coverage_counts = {}
        for c in coverages:
            coverage_counts[c] = coverage_counts.get(c, 0) + 1

        axes[1, 0].bar(coverage_counts.keys(), coverage_counts.values())
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Focus Range Coverage')

        # Alignment quality
        alignments = [r.get('input_analysis', {}).get(
            'alignment_quality', {}).get('quality', 'poor') for r in test_results]
        alignment_counts = {}
        for a in alignments:
            alignment_counts[a] = alignment_counts.get(a, 0) + 1

        axes[1, 1].bar(alignment_counts.keys(), alignment_counts.values())
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Alignment Quality')

        plt.tight_layout()
        plt.savefig(output_dir / 'test_report.png',
                    dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main testing function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Focus-Stacking Pipeline Tester")
    parser.add_argument('input_dir', type=Path,
                        help='Directory containing image sets')
    parser.add_argument('output_dir', type=Path,
                        help='Output directory for results')
    parser.add_argument('--pattern', default='*.tiff',
                        help='Image file pattern')
    parser.add_argument('--group-by', default='angle', choices=['angle', 'directory'],
                        help='How to group images')

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Input directory does not exist: {args.input_dir}")
        return

    # Initialize tester
    tester = PipelineTester()

    # Find image sets
    if args.group_by == 'angle':
        # Group by angle (assuming filename pattern: angle_001_focus_001.tiff)
        image_sets = {}
        for file_path in args.input_dir.glob(args.pattern):
            parts = file_path.stem.split('_')
            if len(parts) >= 2:
                angle_key = f"{parts[0]}_{parts[1]}"
                if angle_key not in image_sets:
                    image_sets[angle_key] = []
                image_sets[angle_key].append(file_path)
    else:
        # Group by directory
        image_sets = {}
        for subdir in args.input_dir.iterdir():
            if subdir.is_dir():
                files = list(subdir.glob(args.pattern))
                if files:
                    image_sets[subdir.name] = sorted(files)

    if not image_sets:
        print(f"No image sets found in {args.input_dir}")
        return

    print(f"Found {len(image_sets)} image sets to test")

    # Test each set
    test_results = []
    for set_name, image_paths in image_sets.items():
        print(f"\nTesting set: {set_name} ({len(image_paths)} images)")

        output_path = args.output_dir / f"{set_name}_test_result.tiff"
        result = tester.test_focus_stacking(image_paths, output_path)

        # Add set info
        result['set_name'] = set_name
        result['image_paths'] = [str(p) for p in image_paths]

        test_results.append(result)

        # Print summary
        if result['success']:
            print(f"  ✅ Success - {result['processing_time']:.1f}s")
            print(
                f"  Suitability: {result['input_analysis']['suitability_score']:.1f}/100")
        else:
            print(f"  ❌ Failed - {result.get('error', 'Unknown error')}")

    # Generate report
    summary = tester.generate_test_report(test_results, args.output_dir)

    print(f"\n=== Test Summary ===")
    print(f"Total sets: {summary['total_tests']}")
    print(f"Successful: {summary['successful_tests']}")
    print(
        f"Average processing time: {summary['average_processing_time']:.1f}s")
    print(
        f"Average suitability score: {summary['average_suitability_score']:.1f}/100")
    print(f"\nDetailed results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
