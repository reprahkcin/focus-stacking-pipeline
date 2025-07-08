#!/usr/bin/env python3
"""
Raspberry Pi Optimization Script
Optimized settings and utilities for Raspberry Pi HQ Camera focus-stacking
"""

import psutil
import os
import gc
import logging
from pathlib import Path
import yaml
from focus_stack import FocusStacker


class RaspberryPiOptimizer:
    """Optimization utilities for Raspberry Pi hardware"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.memory_limit_gb = 2  # Conservative for Pi 4
        cpu_count = psutil.cpu_count()
        # Pi 4 has 4 cores
        self.cpu_cores = min(4, cpu_count if cpu_count else 1)

    def check_system_resources(self) -> dict:
        """Check available system resources"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)

        info = {
            'total_memory_gb': memory.total / (1024**3),
            'available_memory_gb': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'cpu_percent': cpu_percent,
            'cpu_cores': self.cpu_cores,
            'temperature': self._get_cpu_temperature()
        }

        self.logger.info(f"System resources: {info}")
        return info

    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature (Raspberry Pi specific)"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read()) / 1000.0
            return temp
        except:
            return 0.0

    def optimize_config_for_pi(self, config_path: str = "config.yaml") -> str:
        """Create Pi-optimized configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Memory optimization
        config['processing']['memory_limit_gb'] = self.memory_limit_gb
        config['processing']['batch_size'] = 3  # Smaller batches
        config['processing']['parallel_workers'] = 2  # Conservative threading

        # Performance settings
        config['performance']['use_gpu'] = False  # Pi doesn't have CUDA
        config['performance']['cache_intermediate'] = False  # Save memory
        config['performance']['cleanup_temp'] = True

        # Focus stacking optimization
        config['focus_stacking']['kernel_size'] = 3  # Smaller kernel
        # Fewer iterations
        config['focus_stacking']['alignment']['max_iterations'] = 30

        # Output optimization
        config['output']['compression'] = 'lzw'  # Good compression
        config['output']['quality'] = 90  # Slightly lower quality for speed

        # Create optimized config
        pi_config_path = "config_pi.yaml"
        with open(pi_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        self.logger.info(f"Pi-optimized config saved to: {pi_config_path}")
        return pi_config_path

    def monitor_resources(self, interval: int = 30):
        """Monitor system resources during processing"""
        import time
        import threading

        def monitor():
            while True:
                resources = self.check_system_resources()

                # Warn if memory usage is high
                if resources['memory_percent'] > 80:
                    self.logger.warning(
                        f"High memory usage: {resources['memory_percent']:.1f}%")

                # Warn if CPU temperature is high
                if resources['temperature'] > 70:
                    self.logger.warning(
                        f"High CPU temperature: {resources['temperature']:.1f}°C")

                time.sleep(interval)

        # Start monitoring in background
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        return monitor_thread

    def cleanup_memory(self):
        """Force garbage collection and memory cleanup"""
        gc.collect()

        # Clear OpenCV cache if possible
        try:
            import cv2
            cv2.destroyAllWindows()
        except:
            pass

        self.logger.info("Memory cleanup completed")


class PiFocusStacker(FocusStacker):
    """Raspberry Pi optimized focus stacker"""

    def __init__(self, config_path: str = "config_pi.yaml"):
        super().__init__(config_path)
        self.optimizer = RaspberryPiOptimizer()
        self.monitor_thread = None

    def process_with_monitoring(self, input_dir: Path, output_dir: Path) -> dict:
        """Process with resource monitoring"""
        # Start resource monitoring
        self.monitor_thread = self.optimizer.monitor_resources()

        try:
            # Check initial resources
            self.optimizer.check_system_resources()

            # Process images
            results = self.batch_process(input_dir, output_dir)

            # Final cleanup
            self.optimizer.cleanup_memory()

            return results

        finally:
            # Stop monitoring
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1)

    def create_focus_stack(self, images, align: bool = True):
        """Override with Pi-specific optimizations"""
        # Check memory before processing
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            self.logger.warning("High memory usage, forcing cleanup")
            self.optimizer.cleanup_memory()

        # Process with parent method
        result = super().create_focus_stack(images, align)

        # Cleanup after processing
        self.optimizer.cleanup_memory()

        return result


def create_pi_optimized_workflow():
    """Create a complete Pi-optimized workflow"""

    # Initialize optimizer
    optimizer = RaspberryPiOptimizer()

    # Check system
    resources = optimizer.check_system_resources()

    print("=== Raspberry Pi Focus-Stacking Setup ===")
    print(f"Available memory: {resources['available_memory_gb']:.1f} GB")
    print(f"CPU cores: {resources['cpu_cores']}")
    print(f"CPU temperature: {resources['temperature']:.1f}°C")

    # Create optimized config
    pi_config = optimizer.optimize_config_for_pi()

    print(f"\nOptimized configuration created: {pi_config}")
    print("\nRecommended workflow:")
    print("1. Place images in input_images/")
    print("2. Run: python examples/raspberry_pi_optimization.py process")
    print("3. Monitor system resources during processing")

    return pi_config


def process_images_pi():
    """Process images with Pi optimizations"""
    from pathlib import Path

    # Create optimized config
    optimizer = RaspberryPiOptimizer()
    pi_config = optimizer.optimize_config_for_pi()

    # Initialize Pi-optimized stacker
    stacker = PiFocusStacker(pi_config)

    # Process with monitoring
    input_dir = Path("input_images")
    output_dir = Path("output_stacks_pi")

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        print("Please place your images in the input_images/ directory")
        return

    print("Starting Pi-optimized focus-stacking...")
    results = stacker.process_with_monitoring(input_dir, output_dir)

    # Report results
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    print(f"\nProcessing complete: {successful}/{total} groups successful")

    if successful < total:
        failed = [name for name, success in results.items() if not success]
        print(f"Failed groups: {', '.join(failed)}")


def main():
    """Main function for Pi optimization"""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python raspberry_pi_optimization.py setup    # Create optimized config")
        print("  python raspberry_pi_optimization.py process  # Process images")
        return

    command = sys.argv[1]

    if command == "setup":
        create_pi_optimized_workflow()
    elif command == "process":
        process_images_pi()
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
