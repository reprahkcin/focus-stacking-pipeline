#!/usr/bin/env python3
"""
Quick Test Script for Focus-Stacking Pipeline
Simple validation without heavy dependencies
"""

import os
import sys
from pathlib import Path
import json
import time


def check_dependencies():
    """Check if required dependencies are available"""
    missing = []

    try:
        import cv2
        print("✅ OpenCV available")
    except ImportError:
        missing.append("opencv-python")
        print("❌ OpenCV not available")

    try:
        import numpy
        print("✅ NumPy available")
    except ImportError:
        missing.append("numpy")
        print("❌ NumPy not available")

    try:
        import yaml
        print("✅ PyYAML available")
    except ImportError:
        missing.append("pyyaml")
        print("❌ PyYAML not available")

    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    return True


def validate_image_files(image_dir: Path) -> dict:
    """Validate image files in directory"""
    print(f"\nValidating images in: {image_dir}")

    if not image_dir.exists():
        return {"error": f"Directory does not exist: {image_dir}"}

    # Find image files
    image_extensions = ['.tiff', '.tif', '.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        return {"error": f"No image files found in {image_dir}"}

    # Analyze files
    total_size = 0
    dimensions = []
    file_info = []

    for file_path in image_files:
        try:
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            total_size += file_size

            # Try to get image dimensions
            try:
                import cv2
                img = cv2.imread(str(file_path))
                if img is not None:
                    dimensions.append(img.shape[:2])
                    file_info.append({
                        "name": file_path.name,
                        "size_mb": file_size,
                        "dimensions": img.shape[:2]
                    })
                else:
                    file_info.append({
                        "name": file_path.name,
                        "size_mb": file_size,
                        "dimensions": "failed_to_load"
                    })
            except:
                file_info.append({
                    "name": file_path.name,
                    "size_mb": file_size,
                    "dimensions": "unknown"
                })

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Group by potential angle sets
    angle_groups = {}
    for file_path in image_files:
        parts = file_path.stem.split('_')
        if len(parts) >= 2:
            angle_key = f"{parts[0]}_{parts[1]}"
            if angle_key not in angle_groups:
                angle_groups[angle_key] = []
            angle_groups[angle_key].append(file_path.name)

    return {
        "total_files": len(image_files),
        "total_size_mb": total_size,
        "file_info": file_info,
        "angle_groups": angle_groups,
        "unique_dimensions": list(set(dimensions)) if dimensions else []
    }


def test_basic_functionality():
    """Test basic pipeline functionality"""
    print("\nTesting basic functionality...")

    # Check if main modules can be imported
    try:
        from focus_stack import FocusStacker
        print("✅ FocusStacker can be imported")
    except Exception as e:
        print(f"❌ FocusStacker import failed: {e}")
        return False

    # Check if config exists
    if Path("config.yaml").exists():
        print("✅ Configuration file exists")
    else:
        print("❌ Configuration file missing")
        return False

    return True


def run_quick_test(image_dir: Path, output_dir: Path):
    """Run a quick test with available images"""
    print(f"\nRunning quick test...")

    # Validate images
    validation = validate_image_files(image_dir)
    if "error" in validation:
        print(f"Validation failed: {validation['error']}")
        return

    print(f"Found {validation['total_files']} images")
    print(f"Total size: {validation['total_size_mb']:.1f} MB")
    print(f"Angle groups: {len(validation['angle_groups'])}")

    # Test with first group if available
    if validation['angle_groups']:
        first_group = list(validation['angle_groups'].keys())[0]
        group_files = validation['angle_groups'][first_group]

        print(
            f"\nTesting with group: {first_group} ({len(group_files)} images)")

        # Create image paths
        image_paths = [image_dir / filename for filename in group_files]

        # Test focus stacking
        try:
            from focus_stack import FocusStacker

            start_time = time.time()
            stacker = FocusStacker()

            output_path = output_dir / f"{first_group}_quick_test.tiff"
            success = stacker.process_image_set(image_paths, output_path)

            processing_time = time.time() - start_time

            if success:
                print(f"✅ Quick test successful!")
                print(f"Processing time: {processing_time:.1f}s")
                print(f"Output saved to: {output_path}")

                # Check output
                if output_path.exists():
                    output_size = output_path.stat().st_size / (1024 * 1024)
                    print(f"Output size: {output_size:.1f} MB")
                else:
                    print("❌ Output file not found")
            else:
                print("❌ Quick test failed")

        except Exception as e:
            print(f"❌ Quick test error: {e}")
    else:
        print("No angle groups found for testing")


def main():
    """Main test function"""
    if len(sys.argv) < 2:
        print(
            "Usage: python quick_test.py <image_directory> [output_directory]")
        print("\nExample:")
        print("  python quick_test.py ./test_images ./test_results")
        return

    image_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(
        sys.argv) > 2 else Path("./test_results")

    print("Focus-Stacking Pipeline - Quick Test")
    print("=" * 40)

    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies first.")
        return

    # Test basic functionality
    if not test_basic_functionality():
        print("\nBasic functionality test failed.")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run quick test
    run_quick_test(image_dir, output_dir)

    print(f"\nQuick test complete! Check {output_dir} for results.")


if __name__ == "__main__":
    main()
