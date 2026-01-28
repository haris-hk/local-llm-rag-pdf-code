@echo off
echo Running preprocess.py with CUDA-enabled PyTorch from conda environment...
conda run -n base python preprocess.py
pause
