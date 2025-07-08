#!/usr/bin/env python3
"""
Focus-Stacking Pipeline for Photogrammetry
Optimized for Raspberry Pi HQ Camera and large image sets
"""

import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FocusStackConfig:
    """Configuration for focus stacking process"""
    method: str = "laplacian"
    kernel_size: int = 3
    threshold: float = 0.01
    blend_mode: str = "weighted"
    max_shift: int = 50
    confidence_threshold: float = 0.8
    batch_size: int = 10
    parallel_workers: int = 4


class FocusStacker:
    """
    Advanced focus-stacking implementation optimized for photogrammetry
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the focus stacker with configuration"""
        self.config = self._load_config(config_path)
        self.focus_config = FocusStackConfig(**self.config['focus_stacking'])
        self._setup_logging()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config['logging']
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_config['file']),
                logging.StreamHandler(
                ) if log_config['console_output'] else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def calculate_sharpness(self, image: np.ndarray, method: str = "laplacian") -> np.ndarray:
        """
        Calculate sharpness map using various methods
        Optimized for photogrammetry accuracy
        """
        if method == "laplacian":
            # Laplacian variance - most reliable for focus detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
                image.shape) == 3 else image
            laplacian = cv2.Laplacian(
                gray, cv2.CV_64F, ksize=self.focus_config.kernel_size)
            return np.abs(laplacian)

        elif method == "sobel":
            # Sobel gradient magnitude
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
                image.shape) == 3 else image
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(sobelx**2 + sobely**2)

        elif method == "variance":
            # Local variance - good for texture-rich areas
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
                image.shape) == 3 else image
            kernel = np.ones((5, 5), np.float32) / 25
            mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            variance = cv2.filter2D(
                (gray.astype(np.float32) - mean)**2, -1, kernel)
            return variance

        elif method == "tenengrad":
            # Tenengrad operator - edge-based sharpness
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
                image.shape) == 3 else image
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            tenengrad = np.sqrt(sobelx**2 + sobely**2)
            return tenengrad

        else:
            raise ValueError(f"Unknown sharpness method: {method}")

    def align_images(self, images: List[np.ndarray], reference_idx: int = 0) -> List[np.ndarray]:
        """
        Align images using Enhanced Correlation Coefficient (ECC)
        Critical for photogrammetry accuracy
        """
        if len(images) <= 1:
            return images

        reference = images[reference_idx]
        aligned_images = [reference]

        # Convert to grayscale for alignment
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if len(
            reference.shape) == 3 else reference

        for i, image in enumerate(images):
            if i == reference_idx:
                continue

            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
                image.shape) == 3 else image

            # Initialize transformation matrix
            warp_matrix = np.eye(2, 3, dtype=np.float32)

            # Define termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        self.config['focus_stacking']['alignment']['max_iterations'],
                        self.config['focus_stacking']['alignment']['epsilon'])

            try:
                # Find transformation matrix
                _, warp_matrix = cv2.findTransformECC(
                    ref_gray, img_gray, warp_matrix,
                    cv2.MOTION_EUCLIDEAN, criteria
                )

                # Apply transformation
                aligned = cv2.warpAffine(
                    image, warp_matrix, (image.shape[1], image.shape[0]))

                # Check if alignment was successful (not too much shift)
                shift_magnitude = np.sqrt(
                    warp_matrix[0, 2]**2 + warp_matrix[1, 2]**2)
                if shift_magnitude > self.focus_config.max_shift:
                    self.logger.warning(
                        f"Large shift detected ({shift_magnitude:.2f}px) for image {i}")
                    aligned = image  # Use original if shift too large

                aligned_images.append(aligned)

            except cv2.error as e:
                self.logger.warning(f"Alignment failed for image {i}: {e}")
                aligned_images.append(image)

        return aligned_images

    def create_focus_stack(self, images: List[np.ndarray],
                           align: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create focus-stacked image with quality map
        Returns stacked image and confidence map
        """
        if not images:
            raise ValueError("No images provided for focus stacking")

        self.logger.info(f"Processing {len(images)} images for focus stacking")

        # Align images if requested
        if align and len(images) > 1:
            self.logger.info("Aligning images...")
            images = self.align_images(images)

        # Calculate sharpness maps
        self.logger.info("Calculating sharpness maps...")
        sharpness_maps = []
        for img in tqdm(images, desc="Calculating sharpness"):
            sharpness = self.calculate_sharpness(img, self.focus_config.method)
            sharpness_maps.append(sharpness)

        # Normalize sharpness maps
        sharpness_maps = np.array(sharpness_maps)
        sharpness_maps = (sharpness_maps - sharpness_maps.min()) / \
            (sharpness_maps.max() - sharpness_maps.min() + 1e-8)

        # Create focus stack
        self.logger.info("Creating focus stack...")
        if self.focus_config.blend_mode == "weighted":
            # Weighted blending based on sharpness
            weights = sharpness_maps / \
                (np.sum(sharpness_maps, axis=0, keepdims=True) + 1e-8)
            stacked = np.sum(images * weights[:, :, :, np.newaxis], axis=0)
            confidence = np.max(sharpness_maps, axis=0)

        elif self.focus_config.blend_mode == "max":
            # Maximum sharpness selection
            max_indices = np.argmax(sharpness_maps, axis=0)
            stacked = np.zeros_like(images[0])
            for i in range(images[0].shape[0]):
                for j in range(images[0].shape[1]):
                    stacked[i, j] = images[max_indices[i, j]][i, j]
            confidence = np.max(sharpness_maps, axis=0)

        else:  # average
            stacked = np.mean(images, axis=0)
            confidence = np.mean(sharpness_maps, axis=0)

        # Convert to appropriate data type
        stacked = np.clip(stacked, 0, 255).astype(np.uint8)
        confidence = np.clip(confidence, 0, 1).astype(np.float32)

        return stacked, confidence

    def process_image_set(self, image_paths: List[Path],
                          output_path: Path,
                          save_confidence: bool = True) -> bool:
        """
        Process a set of images and save the focus-stacked result
        """
        try:
            # Load images
            self.logger.info(f"Loading {len(image_paths)} images...")
            images = []
            for path in tqdm(image_paths, desc="Loading images"):
                img = cv2.imread(str(path))
                if img is None:
                    self.logger.error(f"Failed to load image: {path}")
                    return False
                images.append(img)

            # Create focus stack
            stacked, confidence = self.create_focus_stack(images, align=True)

            # Save results
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), stacked)

            if save_confidence:
                confidence_path = output_path.with_suffix('.confidence.tiff')
                cv2.imwrite(str(confidence_path),
                            (confidence * 255).astype(np.uint8))

            self.logger.info(f"Focus stack saved to: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error processing image set: {e}")
            return False

    def batch_process(self, input_dir: Path, output_dir: Path,
                      group_by: str = "angle") -> Dict[str, bool]:
        """
        Process multiple image sets in batch
        """
        results = {}

        # Group images by angle or other criteria
        if group_by == "angle":
            # Assuming filename pattern: angle_001_focus_001.tiff
            image_groups = self._group_by_angle(input_dir)
        else:
            # Default grouping by directory
            image_groups = self._group_by_directory(input_dir)

        self.logger.info(f"Found {len(image_groups)} image groups to process")

        # Process groups
        for group_name, image_paths in image_groups.items():
            output_path = output_dir / f"{group_name}_stacked.tiff"
            self.logger.info(f"Processing group: {group_name}")

            success = self.process_image_set(image_paths, output_path)
            results[group_name] = success

        return results

    def _group_by_angle(self, input_dir: Path) -> Dict[str, List[Path]]:
        """Group images by angle based on filename pattern"""
        pattern = self.config['input']['pattern']
        image_files = list(input_dir.glob(pattern))

        groups = {}
        for file_path in image_files:
            # Extract angle from filename (customize based on your naming convention)
            # Example: angle_001_focus_001.tiff -> angle_001
            parts = file_path.stem.split('_')
            if len(parts) >= 2:
                angle_key = f"{parts[0]}_{parts[1]}"
                if angle_key not in groups:
                    groups[angle_key] = []
                groups[angle_key].append(file_path)

        # Sort images within each group
        for group in groups.values():
            group.sort()

        return groups

    def _group_by_directory(self, input_dir: Path) -> Dict[str, List[Path]]:
        """Group images by subdirectory"""
        groups = {}
        for subdir in input_dir.iterdir():
            if subdir.is_dir():
                pattern = self.config['input']['pattern']
                image_files = list(subdir.glob(pattern))
                if image_files:
                    groups[subdir.name] = sorted(image_files)
        return groups

    def _load_image(self, path: Path) -> Optional[np.ndarray]:
        """Load a single image with error handling"""
        try:
            img = cv2.imread(str(path))
            if img is None:
                self.logger.error(f"Failed to load image: {path}")
                return None
            return img
        except Exception as e:
            self.logger.error(f"Error loading image {path}: {e}")
            return None

    def _save_image(self, path: Path, image: np.ndarray) -> bool:
        """Save an image with error handling"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(path), image)
            if not success:
                self.logger.error(f"Failed to save image: {path}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error saving image {path}: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    stacker = FocusStacker()

    # Process a single set of images
    # image_paths = [Path("input/angle_001_focus_001.tiff"), ...]
    # success = stacker.process_image_set(image_paths, Path("output/angle_001_stacked.tiff"))

    # Process all images in batch
    # results = stacker.batch_process(Path("input"), Path("output"))
