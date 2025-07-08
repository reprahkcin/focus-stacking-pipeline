#!/usr/bin/env python3
"""
Setup script for remote and GPU processing
"""

import yaml
import subprocess
import sys
from pathlib import Path


def check_gpu_availability():
    """Check if CUDA GPU is available"""
    try:
        import cv2
        gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
        if gpu_count > 0:
            print(f"✅ Found {gpu_count} CUDA device(s)")
            return True
        else:
            print("❌ No CUDA devices found")
            return False
    except ImportError:
        print("❌ OpenCV CUDA support not available")
        return False
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False


def check_ssh_connection(host, user):
    """Test SSH connection to remote machine"""
    try:
        result = subprocess.run(
            ['ssh', '-o', 'ConnectTimeout=10',
                f'{user}@{host}', 'echo "SSH connection successful"'],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            print(f"✅ SSH connection to {user}@{host} successful")
            return True
        else:
            print(f"❌ SSH connection to {user}@{host} failed")
            return False
    except Exception as e:
        print(f"❌ SSH test failed: {e}")
        return False


def setup_gpu_processing():
    """Setup GPU processing configuration"""
    print("\n=== GPU Processing Setup ===")

    if not check_gpu_availability():
        print("GPU processing not available on this machine")
        return False

    # Load current config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Enable GPU processing
    config['performance']['use_gpu'] = True

    # Get GPU device ID
    device_id = input("Enter GPU device ID (default: 0): ").strip()
    if not device_id:
        device_id = 0
    else:
        device_id = int(device_id)

    config['performance']['gpu']['device_id'] = device_id

    # Save updated config
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"✅ GPU processing configured for device {device_id}")
    return True


def setup_remote_processing():
    """Setup remote processing configuration"""
    print("\n=== Remote Processing Setup ===")

    # Get remote machine details
    host = input("Enter remote machine IP/hostname: ").strip()
    user = input("Enter remote username: ").strip()
    remote_path = input(
        "Enter remote path for processing (default: /home/user/focus-stacking): ").strip()

    if not remote_path:
        remote_path = f"/home/{user}/focus-stacking"

    # Test SSH connection
    if not check_ssh_connection(host, user):
        print("Please ensure SSH key authentication is set up")
        return False

    # Load current config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Enable remote processing
    config['performance']['remote_processing']['enabled'] = True
    config['performance']['remote_processing']['remote_host'] = host
    config['performance']['remote_processing']['remote_user'] = user
    config['performance']['remote_processing']['remote_path'] = remote_path

    # Save updated config
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    print(f"✅ Remote processing configured for {user}@{host}:{remote_path}")

    # Instructions for remote setup
    print("\n📋 Next steps for remote machine:")
    print(f"1. SSH to {user}@{host}")
    print(f"2. Create directory: mkdir -p {remote_path}")
    print(f"3. Copy this project to {remote_path}")
    print(f"4. Install dependencies: pip install -r requirements.txt")

    return True


def show_current_config():
    """Show current processing configuration"""
    print("\n=== Current Configuration ===")

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    performance = config.get('performance', {})

    print(
        f"GPU Processing: {'✅ Enabled' if performance.get('use_gpu') else '❌ Disabled'}")
    print(
        f"Remote Processing: {'✅ Enabled' if performance.get('remote_processing', {}).get('enabled') else '❌ Disabled'}")

    if performance.get('remote_processing', {}).get('enabled'):
        remote = performance['remote_processing']
        print(f"  Remote Host: {remote.get('remote_host')}")
        print(f"  Remote User: {remote.get('remote_user')}")
        print(f"  Remote Path: {remote.get('remote_path')}")


def main():
    """Main setup function"""
    print("Focus-Stacking Pipeline - Processing Setup")
    print("=" * 50)

    if not Path('config.yaml').exists():
        print("❌ config.yaml not found. Please run 'python cli.py init' first.")
        return

    while True:
        print("\nOptions:")
        print("1. Show current configuration")
        print("2. Setup GPU processing")
        print("3. Setup remote processing")
        print("4. Test GPU availability")
        print("5. Test remote connection")
        print("6. Exit")

        choice = input("\nEnter your choice (1-6): ").strip()

        if choice == '1':
            show_current_config()
        elif choice == '2':
            setup_gpu_processing()
        elif choice == '3':
            setup_remote_processing()
        elif choice == '4':
            check_gpu_availability()
        elif choice == '5':
            host = input("Enter remote host: ").strip()
            user = input("Enter remote user: ").strip()
            check_ssh_connection(host, user)
        elif choice == '6':
            print("Setup complete!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
