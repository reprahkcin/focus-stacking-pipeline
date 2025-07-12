#!/usr/bin/env python3
"""
Test script for advanced focus stacking using pre-aligned images
"""

import os
import sys
from pathlib import Path
from advanced_focus_stack import AdvancedFocusStacker
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_focus_stack.log'),
        logging.StreamHandler()
    ]
)


def main():
    # Define paths
    test_files_dir = Path("test files")
    stack1_aligned_dir = test_files_dir / "bee mouth parts 01_1" / "aligned"
    stack2_aligned_dir = test_files_dir / "bee mouth parts 02_1" / "align"

    # Create output directory
    output_dir = Path("output_advanced_focus_stacks")
    output_dir.mkdir(exist_ok=True)

    # Initialize advanced focus stacker
    stacker = AdvancedFocusStacker()

    # Process Stack 1 (aligned images)
    print(f"\n{'='*60}")
    print(f"PROCESSING STACK 1 (ALIGNED): {stack1_aligned_dir}")
    print(f"{'='*60}")

    # Get all aligned JPG files from stack 1
    stack1_aligned_images = [f for f in stack1_aligned_dir.glob(
        "*.jpg") if f.is_file() and f.stat().st_size > 0]
    stack1_aligned_images.sort()  # Sort by filename

    print(f"Found {len(stack1_aligned_images)} aligned images in Stack 1")
    print(f"First image: {stack1_aligned_images[0].name}")
    print(f"Last image: {stack1_aligned_images[-1].name}")

    # Process Stack 1 with advanced focus stacking
    output_file1 = output_dir / "bee_mouth_parts_01_1_advanced_stacked.tiff"
    success1 = stacker.process_aligned_stack(
        stack1_aligned_images, output_file1)

    if success1:
        print(f"✓ Stack 1 completed successfully: {output_file1}")
    else:
        print(f"✗ Stack 1 failed")

    # Process Stack 2 (aligned images)
    print(f"\n{'='*60}")
    print(f"PROCESSING STACK 2 (ALIGNED): {stack2_aligned_dir}")
    print(f"{'='*60}")

    # Get all aligned JPG files from stack 2
    stack2_aligned_images = [f for f in stack2_aligned_dir.glob(
        "*.jpg") if f.is_file() and f.stat().st_size > 0]
    stack2_aligned_images.sort()  # Sort by filename

    print(f"Found {len(stack2_aligned_images)} aligned images in Stack 2")
    print(f"First image: {stack2_aligned_images[0].name}")
    print(f"Last image: {stack2_aligned_images[-1].name}")

    # Process Stack 2 with advanced focus stacking
    output_file2 = output_dir / "bee_mouth_parts_02_1_advanced_stacked.tiff"
    success2 = stacker.process_aligned_stack(
        stack2_aligned_images, output_file2)

    if success2:
        print(f"✓ Stack 2 completed successfully: {output_file2}")
    else:
        print(f"✗ Stack 2 failed")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Stack 1: {'✓ SUCCESS' if success1 else '✗ FAILED'}")
    print(f"Stack 2: {'✓ SUCCESS' if success2 else '✗ FAILED'}")

    if success1 and success2:
        print(f"\nBoth stacks processed successfully with advanced focus stacking!")
        print(f"Output files:")
        print(f"  - {output_file1}")
        print(f"  - {output_file2}")
        print(f"  - {output_file1.with_suffix('.confidence.tiff')}")
        print(f"  - {output_file2.with_suffix('.confidence.tiff')}")
    else:
        print(f"\nSome stacks failed. Check the log file for details.")


if __name__ == "__main__":
    main()
