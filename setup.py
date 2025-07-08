#!/usr/bin/env python3
"""
Setup script for Focus-Stacking Pipeline
"""

from setuptools import setup, find_packages
import os

# Read the README file


def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements


def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]


setup(
    name="focus-stacking-pipeline",
    version="1.0.0",
    author="Focus-Stacking Pipeline Team",
    author_email="",
    description="A high-performance focus-stacking pipeline optimized for photogrammetry workflows",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/focus-stacking-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "gpu": [
            "opencv-python-headless>=4.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "focus-stack=cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml"],
    },
)
