#!/usr/bin/env python3
"""
Command-line interface for the Focus-Stacking Pipeline
"""

import click
import yaml
from pathlib import Path
from focus_stack import FocusStacker
import logging


@click.group()
@click.option('--config', '-c', default='config.yaml',
              help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Focus-Stacking Pipeline for Photogrammetry"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose

    # Setup logging
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option('--group-by', default='angle',
              type=click.Choice(['angle', 'directory']),
              help='How to group images for processing')
@click.option('--pattern', default='*.tiff',
              help='File pattern to match')
@click.option('--remote', is_flag=True,
              help='Use remote processing on powerful machine')
@click.option('--gpu', is_flag=True,
              help='Enable GPU acceleration (if available)')
@click.pass_context
def batch(ctx, input_dir, output_dir, group_by, pattern, remote, gpu):
    """Process all image sets in batch mode"""
    config = ctx.obj['config']

    # Update config with command line options
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['input']['pattern'] = pattern

    # Create temporary config
    temp_config = 'temp_config.yaml'
    with open(temp_config, 'w') as f:
        yaml.dump(cfg, f)

    try:
        # Handle remote processing
        if remote:
            from remote_processor import RemoteProcessor
            click.echo("Using remote processing...")
            remote_processor = RemoteProcessor(temp_config)
            results = remote_processor.process_remote(input_dir, output_dir)
            successful = 1 if results.get("success") else 0
            total = 1
        else:
            # Handle GPU processing
            if gpu:
                cfg['performance']['use_gpu'] = True
                with open(temp_config, 'w') as f:
                    yaml.dump(cfg, f)
                click.echo("GPU acceleration enabled")

            stacker = FocusStacker(temp_config)
            results = stacker.batch_process(input_dir, output_dir, group_by)

            # Report results
            successful = sum(1 for success in results.values() if success)
            total = len(results)

        click.echo(
            f"\nProcessing complete: {successful}/{total} groups successful")

        if successful < total and not remote:
            failed = [name for name, success in results.items() if not success]
            click.echo(f"Failed groups: {', '.join(failed)}")

    finally:
        # Cleanup temp config
        Path(temp_config).unlink(missing_ok=True)


@cli.command()
@click.argument('image_files', nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.argument('output_file', type=click.Path(path_type=Path))
@click.option('--method', default='laplacian',
              type=click.Choice(
                  ['laplacian', 'sobel', 'variance', 'tenengrad']),
              help='Sharpness detection method')
@click.option('--blend-mode', default='weighted',
              type=click.Choice(['weighted', 'max', 'average']),
              help='Blending mode for focus stacking')
@click.option('--no-align', is_flag=True,
              help='Skip image alignment')
@click.pass_context
def stack(ctx, image_files, output_file, method, blend_mode, no_align):
    """Focus-stack a single set of images"""
    if not image_files:
        click.echo("Error: No image files specified", err=True)
        return

    config = ctx.obj['config']

    # Update config with command line options
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['focus_stacking']['method'] = method
    cfg['focus_stacking']['blend_mode'] = blend_mode

    # Create temporary config
    temp_config = 'temp_config.yaml'
    with open(temp_config, 'w') as f:
        yaml.dump(cfg, f)

    try:
        stacker = FocusStacker(temp_config)

        # Load images
        images = []
        for path in image_files:
            img = stacker._load_image(path)
            if img is not None:
                images.append(img)

        if not images:
            click.echo("Error: No valid images loaded", err=True)
            return

        # Create focus stack
        stacked, confidence = stacker.create_focus_stack(
            images, align=not no_align)

        # Save results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        stacker._save_image(output_file, stacked)

        # Save confidence map
        confidence_file = output_file.with_suffix('.confidence.tiff')
        stacker._save_image(
            confidence_file, (confidence * 255).astype('uint8'))

        click.echo(f"Focus stack saved to: {output_file}")
        click.echo(f"Confidence map saved to: {confidence_file}")

    finally:
        # Cleanup temp config
        Path(temp_config).unlink(missing_ok=True)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, path_type=Path))
@click.option('--pattern', default='*.tiff',
              help='File pattern to match')
def analyze(ctx, input_dir, pattern):
    """Analyze image sets and show statistics"""
    input_path = Path(input_dir)
    image_files = list(input_path.glob(pattern))

    if not image_files:
        click.echo(f"No images found matching pattern: {pattern}")
        return

    click.echo(f"Found {len(image_files)} images")

    # Group by angle
    groups = {}
    for file_path in image_files:
        parts = file_path.stem.split('_')
        if len(parts) >= 2:
            angle_key = f"{parts[0]}_{parts[1]}"
            if angle_key not in groups:
                groups[angle_key] = []
            groups[angle_key].append(file_path)

    click.echo(f"\nImage groups by angle:")
    for group_name, files in sorted(groups.items()):
        click.echo(f"  {group_name}: {len(files)} images")

    # Show file sizes
    total_size = sum(f.stat().st_size for f in image_files)
    click.echo(f"\nTotal size: {total_size / (1024**3):.2f} GB")


@cli.command()
@click.argument('config_file', type=click.Path(path_type=Path))
def validate(ctx, config_file):
    """Validate configuration file"""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Test configuration
        stacker = FocusStacker(str(config_file))
        click.echo("Configuration is valid!")

        # Show key settings
        click.echo(f"\nKey settings:")
        click.echo(f"  Focus method: {config['focus_stacking']['method']}")
        click.echo(f"  Blend mode: {config['focus_stacking']['blend_mode']}")
        click.echo(
            f"  Alignment: {config['focus_stacking']['alignment']['method']}")
        click.echo(f"  Input pattern: {config['input']['pattern']}")
        click.echo(f"  Output format: {config['output']['format']}")

    except Exception as e:
        click.echo(f"Configuration error: {e}", err=True)


@cli.command()
def init():
    """Initialize a new project with sample configuration"""
    if Path('config.yaml').exists():
        click.echo("config.yaml already exists. Use --force to overwrite.")
        return

    # Create sample config
    sample_config = {
        'input': {
            'format': 'tiff',
            'directory': './input_images',
            'pattern': '*.tiff'
        },
        'output': {
            'directory': './output_stacks',
            'format': 'tiff',
            'quality': 95,
            'compression': 'lzw'
        },
        'camera': {
            'sensor_size': [4056, 3040],
            'pixel_size': 1.55,
            'bit_depth': 12
        },
        'focus_stacking': {
            'method': 'laplacian',
            'kernel_size': 3,
            'threshold': 0.01,
            'blend_mode': 'weighted',
            'alignment': {
                'method': 'ecc',
                'max_iterations': 50,
                'epsilon': 1e-6,
                'warp_mode': 'euclidean'
            },
            'quality_control': {
                'min_sharpness': 0.1,
                'max_shift': 50,
                'outlier_rejection': True,
                'confidence_threshold': 0.8
            }
        },
        'processing': {
            'batch_size': 10,
            'parallel_workers': 4,
            'memory_limit_gb': 8,
            'preprocessing': {
                'denoise': True,
                'denoise_strength': 0.1,
                'normalize': True,
                'equalize_histogram': False
            },
            'postprocessing': {
                'sharpen': False,
                'color_correction': True,
                'lens_correction': True
            }
        },
        'photogrammetry': {
            'colmap': {
                'feature_type': 'sift',
                'max_features': 8192,
                'quality': 'high'
            },
            'output_prep': {
                'resize_factor': 1.0,
                'format': 'tiff',
                'compression': 'lzw',
                'metadata': True
            }
        },
        'logging': {
            'level': 'INFO',
            'file': 'focus_stack.log',
            'console_output': True
        },
        'performance': {
            'use_gpu': False,
            'cache_intermediate': True,
            'cleanup_temp': True
        }
    }

    with open('config.yaml', 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)

    # Create directories
    Path('input_images').mkdir(exist_ok=True)
    Path('output_stacks').mkdir(exist_ok=True)

    click.echo("Project initialized!")
    click.echo("Created:")
    click.echo("  - config.yaml (sample configuration)")
    click.echo("  - input_images/ (place your images here)")
    click.echo("  - output_stacks/ (results will be saved here)")
    click.echo("\nNext steps:")
    click.echo("  1. Place your images in input_images/")
    click.echo("  2. Adjust config.yaml as needed")
    click.echo("  3. Run: python cli.py batch input_images output_stacks")


if __name__ == '__main__':
    cli()
