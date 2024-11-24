#!/bin/bash

conda create --name fyp python=3.12 --yes
eval "$(conda shell.bash hook)"
conda activate fyp
pip install torch_geometric


if nvidia-smi > /dev/null 2>&1; then
    echo "NVIDIA driver version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install 'transformers'
else
    echo "nvidia-smi is not found. Install CPU version of PyTorch Geometric."
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
    pip install torch
    pip install 'transformers[torch]'
fi

pip install stanza editdistance jsonpickle tensorboard
pip install -U scikit-learn datasets

if [ ! -d "data" ]; then
    mkdir data
fi
