#!/usr/bin/env python3
"""
Advanced Focus Stacking Implementation
Uses multiple sharpness detection methods and robust blending
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
import logging


class AdvancedFocusStacker:
    """
    Advanced focus stacking with multiple sharpness detection methods
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_sharpness_laplacian(self, image: np.ndarray) -> np.ndarray:
        """Laplacian variance sharpness detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
            image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        return np.abs(laplacian)

    def calculate_sharpness_sobel(self, image: np.ndarray) -> np.ndarray:
        """Sobel gradient magnitude sharpness detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
            image.shape) == 3 else image
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx**2 + sobely**2)

    def calculate_sharpness_variance(self, image: np.ndarray) -> np.ndarray:
        """Local variance sharpness detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(
            image.shape) == 3 else image
        kernel = np.ones((5, 5), np.float32) / 25
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        variance = cv2.filter2D(
            (gray.astype(np.float32) - mean)**2, -1, kernel)
        return variance

    def calculate_combined_sharpness(self, image: np.ndarray) -> np.ndarray:
        """Combine multiple sharpness detection methods"""
        laplacian = self.calculate_sharpness_laplacian(image)
        sobel = self.calculate_sharpness_sobel(image)
        variance = self.calculate_sharpness_variance(image)

        # Normalize each method
        laplacian = (laplacian - laplacian.min()) / \
            (laplacian.max() - laplacian.min() + 1e-8)
        sobel = (sobel - sobel.min()) / (sobel.max() - sobel.min() + 1e-8)
        variance = (variance - variance.min()) / \
            (variance.max() - variance.min() + 1e-8)

        # Combine with weights
        combined = 0.5 * laplacian + 0.3 * sobel + 0.2 * variance
        return combined

    def create_focus_stack_advanced(self, images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced focus stacking with multiple methods
        """
        if not images:
            raise ValueError("No images provided")

        self.logger.info(
            f"Processing {len(images)} images with advanced focus stacking")

        # Calculate sharpness maps for all images
        sharpness_maps = []
        for img in tqdm(images, desc="Calculating sharpness maps"):
            sharpness = self.calculate_combined_sharpness(img)
            sharpness_maps.append(sharpness)

        sharpness_maps = np.array(sharpness_maps)

        # Apply Gaussian smoothing to sharpness maps for better blending
        for i in range(len(sharpness_maps)):
            sharpness_maps[i] = cv2.GaussianBlur(
                sharpness_maps[i], (5, 5), 1.0)

        # Normalize sharpness maps
        sharpness_maps = (sharpness_maps - sharpness_maps.min()) / \
            (sharpness_maps.max() - sharpness_maps.min() + 1e-8)

        # Create focus stack using maximum sharpness selection
        self.logger.info(
            "Creating focus stack using maximum sharpness selection")

        # Find the image with maximum sharpness at each pixel
        max_indices = np.argmax(sharpness_maps, axis=0)

        # Create output image
        height, width = images[0].shape[:2]
        channels = images[0].shape[2] if len(images[0].shape) == 3 else 1

        if channels == 3:
            stacked = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    best_img_idx = max_indices[i, j]
                    stacked[i, j] = images[best_img_idx][i, j]
        else:
            stacked = np.zeros((height, width), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    best_img_idx = max_indices[i, j]
                    stacked[i, j] = images[best_img_idx][i, j]

        # Calculate confidence map
        confidence = np.max(sharpness_maps, axis=0)

        return stacked, confidence

    def process_aligned_stack(self, image_paths: List[Path], output_path: Path) -> bool:
        """
        Process a stack of pre-aligned images
        """
        try:
            # Load images
            self.logger.info(f"Loading {len(image_paths)} aligned images...")
            images = []
            for path in tqdm(image_paths, desc="Loading aligned images"):
                img = cv2.imread(str(path))
                if img is None:
                    self.logger.error(f"Failed to load image: {path}")
                    return False
                images.append(img)

            # Create focus stack
            stacked, confidence = self.create_focus_stack_advanced(images)

            # Save results
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), stacked)

            # Save confidence map
            confidence_path = output_path.with_suffix('.confidence.tiff')
            cv2.imwrite(str(confidence_path),
                        (confidence * 255).astype(np.uint8))

            self.logger.info(f"Advanced focus stack saved to: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error processing aligned stack: {e}")
            return False
