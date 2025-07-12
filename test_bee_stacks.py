#!/usr/bin/env python3
"""
Test script for processing bee mouth parts focus stacks
Starting from the very beginning with original unaligned images
"""

import os
import sys
from pathlib import Path
from focus_stack import FocusStacker
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bee_stacks_test.log'),
        logging.StreamHandler()
    ]
)


def main():
    # Define paths
    test_files_dir = Path("test files")
    stack1_dir = test_files_dir / "bee mouth parts 01_1"
    stack2_dir = test_files_dir / "bee mouth parts 02_1"

    # Create output directory
    output_dir = Path("output_bee_stacks")
    output_dir.mkdir(exist_ok=True)

    # Initialize focus stacker
    stacker = FocusStacker("config.yaml")

    # Process Stack 1
    print(f"\n{'='*60}")
    print(f"PROCESSING STACK 1: {stack1_dir}")
    print(f"{'='*60}")

    # Get all JPG files from stack 1 (excluding aligned folder)
    stack1_images = [f for f in stack1_dir.glob(
        "*.jpg") if f.is_file() and f.stat().st_size > 0]
    stack1_images.sort()  # Sort by filename

    print(f"Found {len(stack1_images)} images in Stack 1")
    print(f"First image: {stack1_images[0].name}")
    print(f"Last image: {stack1_images[-1].name}")

    # Process Stack 1 with alignment disabled
    output_file1 = output_dir / "bee_mouth_parts_01_1_stacked_noalign.tiff"
    success1 = stacker.process_image_set_noalign(stack1_images, output_file1)

    if success1:
        print(f"✓ Stack 1 completed successfully: {output_file1}")
    else:
        print(f"✗ Stack 1 failed")

    # Process Stack 2
    print(f"\n{'='*60}")
    print(f"PROCESSING STACK 2: {stack2_dir}")
    print(f"{'='*60}")

    # Get all JPG files from stack 2 (excluding align folder)
    stack2_images = [f for f in stack2_dir.glob(
        "*.jpg") if f.is_file() and f.stat().st_size > 0]
    stack2_images.sort()  # Sort by filename

    print(f"Found {len(stack2_images)} images in Stack 2")
    print(f"First image: {stack2_images[0].name}")
    print(f"Last image: {stack2_images[-1].name}")

    # Process Stack 2 with alignment disabled
    output_file2 = output_dir / "bee_mouth_parts_02_1_stacked_noalign.tiff"
    success2 = stacker.process_image_set_noalign(stack2_images, output_file2)

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
        print(f"\nBoth stacks processed successfully!")
        print(f"Output files:")
        print(f"  - {output_file1}")
        print(f"  - {output_file2}")
        print(f"  - {output_file1.with_suffix('.confidence.tiff')}")
        print(f"  - {output_file2.with_suffix('.confidence.tiff')}")
    else:
        print(f"\nSome stacks failed. Check the log file for details.")


if __name__ == "__main__":
    main()
