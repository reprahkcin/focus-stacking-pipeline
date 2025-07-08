#!/usr/bin/env python3
"""
Complete Photogrammetry Pipeline Example
Integrates focus-stacking with COLMAP for 3D reconstruction
"""

import subprocess
import sys
from pathlib import Path
import yaml
import logging
from focus_stack import FocusStacker


class PhotogrammetryPipeline:
    """Complete pipeline from focus-stacking to 3D reconstruction"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.focus_stacker = FocusStacker(config_path)
        self._setup_logging()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def run_focus_stacking(self, input_dir: Path, output_dir: Path) -> bool:
        """Run focus-stacking on all image sets"""
        self.logger.info("Starting focus-stacking phase...")

        try:
            results = self.focus_stacker.batch_process(input_dir, output_dir)
            successful = sum(1 for success in results.values() if success)
            total = len(results)

            self.logger.info(
                f"Focus-stacking complete: {successful}/{total} successful")
            return successful > 0

        except Exception as e:
            self.logger.error(f"Focus-stacking failed: {e}")
            return False

    def check_colmap_installation(self) -> bool:
        """Check if COLMAP is installed and accessible"""
        try:
            result = subprocess.run(['colmap', '--version'],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info(f"COLMAP version: {result.stdout.strip()}")
                return True
            else:
                self.logger.error("COLMAP not found or not working")
                return False
        except FileNotFoundError:
            self.logger.error(
                "COLMAP not installed. Please install COLMAP first.")
            return False

    def run_colmap_feature_extraction(self, image_path: Path, database_path: Path) -> bool:
        """Run COLMAP feature extraction"""
        self.logger.info("Running COLMAP feature extraction...")

        cmd = [
            'colmap', 'feature_extractor',
            '--database_path', str(database_path),
            '--image_path', str(image_path),
            '--ImageReader.single_camera', '1',
            '--SiftExtraction.use_gpu', '1' if self.config['performance']['use_gpu'] else '0'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("Feature extraction completed successfully")
                return True
            else:
                self.logger.error(
                    f"Feature extraction failed: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return False

    def run_colmap_feature_matching(self, database_path: Path) -> bool:
        """Run COLMAP feature matching"""
        self.logger.info("Running COLMAP feature matching...")

        cmd = [
            'colmap', 'exhaustive_matcher',
            '--database_path', str(database_path),
            '--SiftMatching.use_gpu', '1' if self.config['performance']['use_gpu'] else '0'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("Feature matching completed successfully")
                return True
            else:
                self.logger.error(f"Feature matching failed: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"Feature matching error: {e}")
            return False

    def run_colmap_sparse_reconstruction(self, database_path: Path,
                                         image_path: Path, output_path: Path) -> bool:
        """Run COLMAP sparse reconstruction"""
        self.logger.info("Running COLMAP sparse reconstruction...")

        cmd = [
            'colmap', 'mapper',
            '--database_path', str(database_path),
            '--image_path', str(image_path),
            '--output_path', str(output_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info(
                    "Sparse reconstruction completed successfully")
                return True
            else:
                self.logger.error(
                    f"Sparse reconstruction failed: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"Sparse reconstruction error: {e}")
            return False

    def run_colmap_dense_reconstruction(self, sparse_path: Path,
                                        image_path: Path, dense_path: Path) -> bool:
        """Run COLMAP dense reconstruction"""
        self.logger.info("Running COLMAP dense reconstruction...")

        # Step 1: Image undistortion
        undistort_cmd = [
            'colmap', 'image_undistorter',
            '--image_path', str(image_path),
            '--input_path', str(sparse_path / '0'),
            '--output_path', str(dense_path),
            '--output_type', 'COLMAP'
        ]

        try:
            result = subprocess.run(
                undistort_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(
                    f"Image undistortion failed: {result.stderr}")
                return False

            # Step 2: Patch match stereo
            stereo_cmd = [
                'colmap', 'patch_match_stereo',
                '--workspace_path', str(dense_path),
                '--workspace_format', 'COLMAP',
                '--PatchMatchStereo.geom_consistency', 'true'
            ]

            result = subprocess.run(stereo_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(
                    f"Patch match stereo failed: {result.stderr}")
                return False

            # Step 3: Stereo fusion
            fusion_cmd = [
                'colmap', 'stereo_fusion',
                '--workspace_path', str(dense_path),
                '--workspace_format', 'COLMAP',
                '--output_path', str(dense_path / 'fused.ply')
            ]

            result = subprocess.run(fusion_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("Dense reconstruction completed successfully")
                return True
            else:
                self.logger.error(f"Stereo fusion failed: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Dense reconstruction error: {e}")
            return False

    def run_complete_pipeline(self, input_dir: Path, output_dir: Path) -> bool:
        """Run the complete photogrammetry pipeline"""
        self.logger.info("Starting complete photogrammetry pipeline...")

        # Create output directories
        focus_stacks_dir = output_dir / "focus_stacks"
        colmap_dir = output_dir / "colmap"
        database_path = colmap_dir / "database.db"
        sparse_path = colmap_dir / "sparse"
        dense_path = colmap_dir / "dense"

        for path in [focus_stacks_dir, colmap_dir, sparse_path, dense_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Step 1: Focus stacking
        if not self.run_focus_stacking(input_dir, focus_stacks_dir):
            self.logger.error("Focus stacking failed. Aborting pipeline.")
            return False

        # Step 2: Check COLMAP installation
        if not self.check_colmap_installation():
            self.logger.error(
                "COLMAP not available. Skipping 3D reconstruction.")
            return False

        # Step 3: COLMAP feature extraction
        if not self.run_colmap_feature_extraction(focus_stacks_dir, database_path):
            self.logger.error("Feature extraction failed. Aborting pipeline.")
            return False

        # Step 4: COLMAP feature matching
        if not self.run_colmap_feature_matching(database_path):
            self.logger.error("Feature matching failed. Aborting pipeline.")
            return False

        # Step 5: COLMAP sparse reconstruction
        if not self.run_colmap_sparse_reconstruction(database_path, focus_stacks_dir, sparse_path):
            self.logger.error(
                "Sparse reconstruction failed. Aborting pipeline.")
            return False

        # Step 6: COLMAP dense reconstruction
        if not self.run_colmap_dense_reconstruction(sparse_path, focus_stacks_dir, dense_path):
            self.logger.error("Dense reconstruction failed.")
            return False

        self.logger.info("Complete pipeline finished successfully!")
        self.logger.info(f"Results saved to: {output_dir}")
        self.logger.info(f"3D model: {dense_path / 'fused.ply'}")

        return True


def main():
    """Example usage of the complete pipeline"""
    if len(sys.argv) != 3:
        print("Usage: python photogrammetry_pipeline.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Run pipeline
    pipeline = PhotogrammetryPipeline()
    success = pipeline.run_complete_pipeline(input_dir, output_dir)

    if success:
        print("Pipeline completed successfully!")
    else:
        print("Pipeline failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
