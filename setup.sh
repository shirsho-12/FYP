#!/bin/bash

conda env create -f environment.yml
conda activate fyp
# pip install torch_geometric

# Does not work, use environment.yml
# if nvidia-smi > /dev/null 2>&1; then
#     echo "NVIDIA driver version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
#     pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
#     pip install torch --index-url https://download.pytorch.org/whl/cu121
#     pip install 'transformers'
# else
#     echo "nvidia-smi is not found. Install CPU version of PyTorch Geometric."
#     pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
#     pip install torch
#     pip install 'transformers[torch]'
# fi

# pip install stanza editdistance jsonpickle tensorboard
# pip install -U scikit-learn datasets

if [ ! -d "data" ]; then
    mkdir data
fi

if [ ! -d "data/squad_v2" ]; then
    mkdir data/squad_v2
fi

wget https://huggingface.co/datasets/GEM/squad_v2/resolve/main/squad_data/train-v2.0.json -O data/squad_v2/train-v2.0.json
wget https://huggingface.co/datasets/GEM/squad_v2/resolve/main/squad_data/dev-v2.0.json -O data/squad_v2/dev-v2.0.json
