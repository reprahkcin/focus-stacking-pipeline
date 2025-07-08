#!/usr/bin/env python3
"""
Remote Processing Module
Offloads focus-stacking to a more powerful machine
"""

import subprocess
import paramiko
import os
import tempfile
import zipfile
import logging
from pathlib import Path
from typing import List, Dict, Any
import yaml
import json
from tqdm import tqdm


class RemoteProcessor:
    """Handles remote processing on a more powerful machine"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.remote_config = self.config['performance']['remote_processing']
        self.logger = logging.getLogger(__name__)
        self.ssh_client = None

    def _load_config(self, config_path: str) -> dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def connect_remote(self) -> bool:
        """Establish SSH connection to remote machine"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(
                paramiko.AutoAddPolicy())

            # Load SSH key
            ssh_key_path = os.path.expanduser(
                self.remote_config['ssh_key_path'])

            self.ssh_client.connect(
                hostname=self.remote_config['remote_host'],
                username=self.remote_config['remote_user'],
                key_filename=ssh_key_path,
                timeout=30
            )

            self.logger.info(
                f"Connected to remote machine: {self.remote_config['remote_host']}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to remote machine: {e}")
            return False

    def disconnect_remote(self):
        """Close SSH connection"""
        if self.ssh_client:
            self.ssh_client.close()
            self.logger.info("Disconnected from remote machine")

    def upload_files(self, local_path: Path, remote_path: str) -> bool:
        """Upload files to remote machine"""
        try:
            sftp = self.ssh_client.open_sftp()

            if local_path.is_file():
                # Upload single file
                sftp.put(str(local_path), remote_path)
            else:
                # Upload directory
                self._upload_directory(sftp, local_path, remote_path)

            sftp.close()
            self.logger.info(f"Uploaded {local_path} to {remote_path}")
            return True

        except Exception as e:
            self.logger.error(f"Upload failed: {e}")
            return False

    def _upload_directory(self, sftp, local_path: Path, remote_path: str):
        """Recursively upload directory"""
        try:
            sftp.mkdir(remote_path)
        except:
            pass  # Directory might already exist

        for item in local_path.iterdir():
            remote_item_path = f"{remote_path}/{item.name}"

            if item.is_file():
                sftp.put(str(item), remote_item_path)
            elif item.is_dir():
                self._upload_directory(sftp, item, remote_item_path)

    def download_files(self, remote_path: str, local_path: Path) -> bool:
        """Download files from remote machine"""
        try:
            sftp = self.ssh_client.open_sftp()

            if self._is_remote_file(sftp, remote_path):
                # Download single file
                local_path.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote_path, str(local_path))
            else:
                # Download directory
                self._download_directory(sftp, remote_path, local_path)

            sftp.close()
            self.logger.info(f"Downloaded {remote_path} to {local_path}")
            return True

        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            return False

    def _is_remote_file(self, sftp, remote_path: str) -> bool:
        """Check if remote path is a file"""
        try:
            sftp.stat(remote_path)
            return True
        except:
            return False

    def _download_directory(self, sftp, remote_path: str, local_path: Path):
        """Recursively download directory"""
        local_path.mkdir(parents=True, exist_ok=True)

        for item in sftp.listdir_attr(remote_path):
            remote_item_path = f"{remote_path}/{item.filename}"
            local_item_path = local_path / item.filename

            if item.st_mode & 0o40000:  # Directory
                self._download_directory(
                    sftp, remote_item_path, local_item_path)
            else:  # File
                sftp.get(remote_item_path, str(local_item_path))

    def execute_remote_command(self, command: str) -> tuple:
        """Execute command on remote machine"""
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(command)

            # Wait for completion
            exit_status = stdout.channel.recv_exit_status()

            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')

            return exit_status, output, error

        except Exception as e:
            self.logger.error(f"Remote command failed: {e}")
            return -1, "", str(e)

    def setup_remote_environment(self) -> bool:
        """Setup remote environment for processing"""
        remote_path = self.remote_config['remote_path']

        # Check if remote directory exists
        exit_status, output, error = self.execute_remote_command(
            f"test -d {remote_path}")

        if exit_status != 0:
            # Create remote directory
            exit_status, output, error = self.execute_remote_command(
                f"mkdir -p {remote_path}")
            if exit_status != 0:
                self.logger.error(
                    f"Failed to create remote directory: {error}")
                return False

        # Check Python and dependencies
        exit_status, output, error = self.execute_remote_command(
            "python3 --version")
        if exit_status != 0:
            self.logger.error("Python3 not available on remote machine")
            return False

        # Check if requirements are installed
        exit_status, output, error = self.execute_remote_command(
            f"cd {remote_path} && python3 -c 'import cv2, numpy'"
        )
        if exit_status != 0:
            self.logger.warning("OpenCV/NumPy not installed on remote machine")
            # Could add automatic installation here

        return True

    def process_remote(self, input_dir: Path, output_dir: Path) -> Dict[str, bool]:
        """Process images on remote machine"""
        if not self.connect_remote():
            return {"error": False}

        try:
            # Setup remote environment
            if not self.setup_remote_environment():
                return {"error": False}

            remote_path = self.remote_config['remote_path']
            remote_input = f"{remote_path}/input_images"
            remote_output = f"{remote_path}/output_stacks"

            # Upload input images
            self.logger.info("Uploading images to remote machine...")
            if not self.upload_files(input_dir, remote_input):
                return {"error": False}

            # Upload configuration
            config_path = Path("config.yaml")
            if config_path.exists():
                self.upload_files(config_path, f"{remote_path}/config.yaml")

            # Execute focus-stacking on remote machine
            self.logger.info("Executing focus-stacking on remote machine...")
            command = f"""
                cd {remote_path} && 
                python3 cli.py batch input_images output_stacks
            """

            exit_status, output, error = self.execute_remote_command(command)

            if exit_status != 0:
                self.logger.error(f"Remote processing failed: {error}")
                return {"error": False}

            # Download results
            self.logger.info("Downloading results from remote machine...")
            if not self.download_files(remote_output, output_dir):
                return {"error": False}

            # Cleanup remote files
            self.execute_remote_command(
                f"rm -rf {remote_input} {remote_output}")

            self.logger.info("Remote processing completed successfully")
            return {"success": True}

        finally:
            self.disconnect_remote()


class GPUProcessor:
    """GPU-accelerated processing using CUDA"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.gpu_config = self.config['performance']['gpu']
        self.logger = logging.getLogger(__name__)
        self.gpu_available = self._check_gpu_availability()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _check_gpu_availability(self) -> bool:
        """Check if CUDA GPU is available"""
        try:
            import cv2
            gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
            if gpu_count > 0:
                self.logger.info(f"Found {gpu_count} CUDA device(s)")
                return True
            else:
                self.logger.warning("No CUDA devices found")
                return False
        except:
            self.logger.warning("OpenCV CUDA support not available")
            return False

    def set_gpu_device(self, device_id: int = None):
        """Set GPU device for processing"""
        if not self.gpu_available:
            return False

        try:
            import cv2
            device_id = device_id or self.gpu_config['device_id']
            cv2.cuda.setDevice(device_id)
            self.logger.info(f"Using GPU device {device_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set GPU device: {e}")
            return False

    def gpu_focus_stack(self, images: List, align: bool = True):
        """GPU-accelerated focus stacking"""
        if not self.gpu_available:
            self.logger.warning("GPU not available, falling back to CPU")
            return self._cpu_focus_stack(images, align)

        try:
            import cv2
            import numpy as np

            # Upload images to GPU
            gpu_images = []
            for img in images:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)
                gpu_images.append(gpu_img)

            # GPU-accelerated sharpness calculation
            sharpness_maps = []
            for gpu_img in gpu_images:
                # Convert to grayscale on GPU
                gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)

                # Laplacian on GPU
                gpu_laplacian = cv2.cuda.Laplacian(
                    gpu_gray, cv2.CV_64F, ksize=3)

                # Download result
                sharpness = gpu_laplacian.download()
                sharpness_maps.append(np.abs(sharpness))

            # Process on CPU (blending is more complex)
            return self._blend_images(images, sharpness_maps)

        except Exception as e:
            self.logger.error(f"GPU processing failed: {e}")
            return self._cpu_focus_stack(images, align)

    def _cpu_focus_stack(self, images: List, align: bool = True):
        """CPU fallback for focus stacking"""
        # This would use the original CPU implementation
        pass

    def _blend_images(self, images: List, sharpness_maps: List):
        """Blend images based on sharpness maps"""
        # Implementation of blending logic
        pass


def create_remote_config(host: str, user: str, remote_path: str, ssh_key: str = None):
    """Create a configuration for remote processing"""
    config = {
        'performance': {
            'remote_processing': {
                'enabled': True,
                'remote_host': host,
                'remote_user': user,
                'remote_path': remote_path,
                'ssh_key_path': ssh_key or "~/.ssh/id_rsa"
            }
        }
    }

    # Update existing config
    with open('config.yaml', 'r') as f:
        existing_config = yaml.safe_load(f)

    existing_config.update(config)

    with open('config.yaml', 'w') as f:
        yaml.dump(existing_config, f, default_flow_style=False, indent=2)

    print(f"Remote processing configured for {user}@{host}:{remote_path}")


def create_gpu_config(device_id: int = 0, memory_fraction: float = 0.8):
    """Create a configuration for GPU processing"""
    config = {
        'performance': {
            'use_gpu': True,
            'gpu': {
                'device_id': device_id,
                'memory_fraction': memory_fraction,
                'precision': 'float32'
            }
        }
    }

    # Update existing config
    with open('config.yaml', 'r') as f:
        existing_config = yaml.safe_load(f)

    existing_config.update(config)

    with open('config.yaml', 'w') as f:
        yaml.dump(existing_config, f, default_flow_style=False, indent=2)

    print(f"GPU processing configured for device {device_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remote/GPU Processing Setup")
    parser.add_argument("--remote", action="store_true",
                        help="Setup remote processing")
    parser.add_argument("--host", help="Remote host IP")
    parser.add_argument("--user", help="Remote username")
    parser.add_argument("--path", help="Remote path")
    parser.add_argument("--gpu", action="store_true",
                        help="Setup GPU processing")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")

    args = parser.parse_args()

    if args.remote:
        if not all([args.host, args.user, args.path]):
            print("Remote setup requires --host, --user, and --path")
        else:
            create_remote_config(args.host, args.user, args.path)

    if args.gpu:
        create_gpu_config(args.device)
