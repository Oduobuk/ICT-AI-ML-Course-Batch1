#!/usr/bin/env python3
"""
Environment Verification Script

This script checks if all required packages are installed and accessible.
"""
import sys
import subprocess
import pkg_resources

def check_package(package_name, min_version=None):
    """Check if a package is installed and meets version requirements."""
    try:
        pkg = pkg_resources.get_distribution(package_name)
        if min_version and pkg.version < min_version:
            return (False, f"{package_name} version {pkg.version} is below required {min_version}")
        return (True, f"{package_name} {pkg.version} ✓")
    except pkg_resources.DistributionNotFound:
        return (False, f"{package_name} is not installed")

def main():
    """Main function to verify the environment."""
    print("=" * 50)
    print("Environment Verification Tool")
    print("=" * 50)
    
    # Check Python version
    py_version = sys.version_info
    print(f"Python {py_version.major}.{py_version.minor}.{py_version.micro} detected")
    
    if py_version < (3, 8):
        print("⚠️  Python 3.8 or higher is recommended")
    
    # Check required packages
    print("\nChecking required packages:")
    print("-" * 30)
    
    required_packages = [
        ("numpy", "1.21.0"),
        ("pandas", "1.3.0"),
        ("matplotlib", "3.4.0"),
        ("scikit-learn", "0.24.0"),
        ("jupyter", "1.0.0"),
        ("notebook", "6.4.0"),
        ("seaborn", "0.11.0"),
        ("ipykernel", "5.5.0")
    ]
    
    all_ok = True
    for pkg, version in required_packages:
        success, message = check_package(pkg, version)
        print(f"- {message}")
        if not success:
            all_ok = False
    
    # Check CUDA/GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\nGPU: {torch.cuda.get_device_name(0)} ✓")
            print(f"CUDA: {torch.version.cuda} ✓")
        else:
            print("\n⚠️  No GPU detected. Some deep learning operations may be slow.")
    except ImportError:
        print("\n⚠️  PyTorch not installed. GPU check skipped.")
    
    # Final status
    print("\n" + "=" * 50)
    if all_ok:
        print("✅ Environment setup is complete and looks good!")
    else:
        print("❌ Some requirements are not met. Please install missing packages.")
        print("Run: pip install -r requirements.txt")
    print("=" * 50)

if __name__ == "__main__":
    main()
