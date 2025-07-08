# Focus-Stacking Pipeline for Photogrammetry

A high-performance focus-stacking pipeline optimized for photogrammetry workflows, specifically designed for Raspberry Pi HQ Camera and large image sets (up to 1000+ images).

## Features

- **Photogrammetry-Optimized**: Preserves geometric accuracy critical for 3D reconstruction
- **Advanced Alignment**: Enhanced Correlation Coefficient (ECC) alignment with outlier rejection
- **Multiple Sharpness Methods**: Laplacian, Sobel, Variance, and Tenengrad operators
- **Batch Processing**: Handle large datasets efficiently with parallel processing
- **Quality Control**: Confidence maps and quality metrics for validation
- **Raspberry Pi HQ Camera**: Optimized settings for 12MP sensor
- **COLMAP Integration**: Ready for photogrammetry pipeline integration

## Installation

### Prerequisites

- Python 3.8+
- OpenCV with CUDA support (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional: Install COLMAP for Photogrammetry

#### Windows
Download from: https://colmap.github.io/

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install colmap
```

#### macOS
```bash
brew install colmap
```

## Quick Start

### 1. Initialize Project
```bash
python cli.py init
```

### 2. Organize Your Images
Place your focus-stack images in the `input_images/` directory with the naming convention:
```
angle_001_focus_001.tiff
angle_001_focus_002.tiff
...
angle_002_focus_001.tiff
angle_002_focus_002.tiff
...
```

### 3. Process All Images
```bash
python cli.py batch input_images output_stacks
```

### 4. Check Results
Focus-stacked images will be saved in `output_stacks/` with confidence maps.

## Usage Examples

### Single Image Set Processing
```bash
python cli.py stack image1.tiff image2.tiff image3.tiff output_stacked.tiff
```

### Batch Processing with Custom Settings
```bash
python cli.py batch input_images output_stacks --group-by angle --pattern "*.tiff"
```

### Analyze Image Sets
```bash
python cli.py analyze input_images
```

### Validate Configuration
```bash
python cli.py validate config.yaml
```

## Configuration

The `config.yaml` file contains all processing parameters. Key settings:

### Focus Stacking
- `method`: Sharpness detection algorithm (laplacian, sobel, variance, tenengrad)
- `blend_mode`: Blending method (weighted, max, average)
- `alignment`: Image alignment settings for photogrammetry accuracy

### Camera Settings
- Optimized for Raspberry Pi HQ Camera (4056×3040, 12-bit)
- Adjustable for other cameras

### Performance
- `parallel_workers`: Number of parallel processes
- `memory_limit_gb`: Memory usage limit
- `use_gpu`: Enable GPU acceleration (if available)

## Photogrammetry Integration

### Complete Pipeline

1. **Focus Stacking** (this pipeline)
   ```bash
   python cli.py batch input_images output_stacks
   ```

2. **COLMAP Processing**
   ```bash
   # Feature extraction
   colmap feature_extractor --database_path database.db --image_path output_stacks
   
   # Feature matching
   colmap exhaustive_matcher --database_path database.db
   
   # Sparse reconstruction
   colmap mapper --database_path database.db --image_path output_stacks --output_path sparse
   
   # Dense reconstruction
   colmap image_undistorter --image_path output_stacks --input_path sparse/0 --output_path dense --output_type COLMAP
   colmap patch_match_stereo --workspace_path dense
   colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply
   ```

### Quality Control

- **Confidence Maps**: Each stacked image includes a confidence map
- **Alignment Validation**: Automatic detection of excessive image shifts
- **Sharpness Metrics**: Quality assessment for each focus stack

## Performance Optimization

### For Large Datasets (1000+ images)

1. **Memory Management**
   ```yaml
   processing:
     batch_size: 5  # Reduce for limited RAM
     memory_limit_gb: 4
   ```

2. **Parallel Processing**
   ```yaml
   processing:
     parallel_workers: 8  # Adjust based on CPU cores
   ```

3. **GPU Acceleration** (if available)
   ```yaml
   performance:
     use_gpu: true
   ```

### Raspberry Pi Optimization

For Raspberry Pi 4 with 4GB RAM:
```yaml
processing:
  batch_size: 3
  parallel_workers: 2
  memory_limit_gb: 2
```

## File Formats

### Input Formats
- **TIFF**: Recommended for maximum quality
- **RAW**: Supported via rawpy
- **JPEG/PNG**: Standard formats

### Output Formats
- **TIFF**: Lossless, preserves metadata
- **JPEG**: Compressed, smaller files
- **PNG**: Lossless, good compression

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce `batch_size` in config
   - Lower `memory_limit_gb`
   - Process smaller groups

2. **Alignment Failures**
   - Check for excessive camera movement
   - Adjust `max_shift` parameter
   - Use `--no-align` for stable setups

3. **Poor Focus Stacking**
   - Try different `method` (laplacian, sobel, etc.)
   - Adjust `threshold` and `kernel_size`
   - Check image quality and focus range

### Logging

Enable verbose logging for debugging:
```bash
python cli.py --verbose batch input_images output_stacks
```

Logs are saved to `focus_stack.log` by default.

## Advanced Usage

### Custom Sharpness Methods

Add custom sharpness detection in `focus_stack.py`:
```python
def calculate_custom_sharpness(self, image):
    # Your custom algorithm here
    pass
```

### Integration with Other Tools

The pipeline can be integrated with:
- **Meshroom** (AliceVision)
- **OpenMVS**
- **RealityCapture** (commercial)
- **Agisoft Metashape** (commercial)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV community for computer vision algorithms
- COLMAP team for photogrammetry tools
- Raspberry Pi Foundation for camera hardware
