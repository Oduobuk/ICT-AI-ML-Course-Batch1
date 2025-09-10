# Development Environment Setup Guide

## 1. System Requirements
- Operating System: Windows 10/11, macOS 10.15+, or Linux
- Minimum 8GB RAM (16GB recommended)
- At least 10GB free disk space
- Python 3.8-3.10

## 2. Installation Steps

### Option A: Using Anaconda (Recommended)
1. Download and install [Anaconda](https://www.anaconda.com/products/distribution)
2. Create a new conda environment:
   ```bash
   conda create -n aiml_course python=3.9
   conda activate aiml_course
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Option B: Using Python venv
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## 3. Jupyter Lab Setup
1. Install Jupyter Lab:
   ```bash
   pip install jupyterlab
   ```
2. Register the kernel:
   ```bash
   python -m ipykernel install --user --name=aiml_course
   ```
3. Launch Jupyter Lab:
   ```bash
   jupyter lab
   ```

## 4. Git Setup
1. Install Git from [git-scm.com](https://git-scm.com/)
2. Configure Git:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```
3. Set up SSH keys (optional but recommended)

## 5. Verify Installation
Run the verification script:
```bash
python scripts/verify_installation.py
```

## 6. Troubleshooting
- For GPU support, install appropriate CUDA drivers
- If you encounter SSL errors, update your certificates
- On Windows, ensure Python is added to PATH during installation
