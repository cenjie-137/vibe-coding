#!/bin/bash
set -e

echo "========================================"
echo " DRIVE Retinal Vessel Segmentation"
echo " AutoDL Deployment Script"
echo "========================================"

PROJECT_DIR="/root/autodl-tmp/medseg_project"
CODE_DIR="/root/autodl-tmp/medseg_project/code"
DATA_DIR="/root/autodl-tmp/medseg_project/data/DRIVE"

echo "1. Creating directory structure..."
mkdir -p ${PROJECT_DIR}/data/DRIVE
mkdir -p ${PROJECT_DIR}/output
mkdir -p ${PROJECT_DIR}/logs

echo "2. Moving code files..."
cp -f *.py ${CODE_DIR}/ || true
cp -f requirements.txt ${CODE_DIR}/ || true

echo "3. Installing dependencies..."
pip install --no-cache-dir -r ${CODE_DIR}/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "4. Checking GPU..."
nvidia-smi

echo "5. Starting training..."
cd ${CODE_DIR}
python -u train.py

echo "========================================"
echo " Training complete!"
echo " Output: ${PROJECT_DIR}/output"
echo " Logs: ${PROJECT_DIR}/logs"
echo "========================================"
