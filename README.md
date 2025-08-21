# Delta Selectivity SAEs

The official repository for the **Delta Selectivity Metric**, a method to evaluate the feature reconstruction capability of Sparse Autoencoders (SAEs) on large language models.

## Overview

This project measures how well SAEs preserve semantic information when reconstructing neural network activations. The key insight is to compare classification performance on binary probing tasks using:

- **Model Selectivity**: Classification accuracy using original model activations
- **SAE Selectivity**: Classification accuracy using SAE-reconstructed activations  
- **Delta Selectivity**: The difference between these two measures

A higher delta selectivity indicates better feature preservation by the SAE.

## Project Structure

```
delta-selectivity-saes/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── main.py                      # Main pipeline with producer-consumer threading
├── compute_selectivity.py       # Alternative GPU-parallel implementation
├── activations.py               # Model forward passes and SAE reconstruction
├── utils.py                     # Utility functions for models, datasets, layers
├── pythia_sae.py               # SAE-specific utilities
├── sparse_tensor.py            # Sparse tensor handling utilities
│
├── explore/                     # Exploratory Jupyter notebooks
│   ├── Explore1.ipynb
│   ├── Explore2.ipynb
│   ├── ExploreActivations.ipynb
│   ├── Explore_cuML.ipynb
│   ├── Torch_cuML.ipynb
│   └── analyze_results.ipynb
│
├── scripts/                     # Batch processing scripts
│   ├── compute_pythia160_*.sh   # Scripts for Pythia-160M experiments
│   ├── compute_pythia410_*.sh   # Scripts for Pythia-410M experiments
│   ├── select_*.sh              # Dataset selection scripts
│   └── *.log                    # Execution logs
│
├── results_160m/                # Results for Pythia-160M model
│   ├── EleutherAI_pythia-160m_wikidata_*.json
│   └── wikidata_160m_logs/
│
├── results_410m/                # Results for Pythia-410M model
│   ├── EleutherAI_pythia-410m_wikidata_*.json
│   └── wikidata_410m_logs/
│
└── *_logs/                      # Various experiment logs
```

## Core Components

### Main Scripts

- **`main.py`**: Primary pipeline using producer-consumer threading for memory efficiency
- **`compute_selectivity.py`**: Alternative implementation with GPU parallelization
- **`activations.py`**: Handles model forward passes and SAE activation reconstruction
- **`utils.py`**: Core utilities for model loading, dataset handling, and probe generation

### Key Features

1. **Multi-GPU Support**: Efficient parallel processing across multiple GPUs
2. **Memory Management**: Producer-consumer pattern to handle large datasets
3. **Flexible Architecture**: Support for different model sizes and SAE configurations
4. **Binary Probing**: Tests on semantic tasks like gender, occupation, politics classification

### Supported Models

- **Pythia-160M**: Smaller model for rapid experimentation
- **Pythia-410M**: Larger model for more comprehensive evaluation
- **SAE Variants**: Various SAE architectures and training configurations

### Datasets

The project evaluates on WikiData-derived binary classification tasks:
- Gender classification
- Occupation prediction  
- Political affiliation
- Athletic status
- Life status (alive/deceased)

## Installation

```bash
pip install -r requirements.txt
```

**Note**: Requires CUDA-compatible GPU for optimal performance with cuML.

## Usage

### Basic Usage

```bash
python main.py \
  --model_name EleutherAI/pythia-160m \
  --sae_name EleutherAI/sae-pythia-160m \
  --datasets 0,1,2 \
  --layers 0,1,2,3 \
  --output_directory ./results \
  --device1 cuda:0 \
  --device2 cuda:1
```

### Parameters

- `--model_name`: Hugging Face model identifier
- `--sae_name`: SAE model identifier
- `--datasets`: Comma-separated dataset indices
- `--layers`: Comma-separated layer indices to analyze
- `--output_directory`: Where to save results
- `--device1/device2`: GPU devices for model and computation
- `--batch_size`: Processing batch size
- `--shuffle_select_n`: Subsample size for efficiency

### Batch Processing

Use scripts in `scripts/` directory for systematic evaluation:

```bash
# Example: Run all Pythia-160M WikiData experiments
./scripts/compute_pythia160_wiki_gender.sh
./scripts/compute_pythia160_wiki_occupation.sh
# ... etc
```

## Results

Results are saved as JSON files containing selectivity metrics:

```json
{
  "layer_0": {
    "mse_pos": 0.123,
    "mse_whole": 0.456, 
    "l0": 12.3,
    "l1": 45.6,
    "gender": {
      "model_sel": 0.789,
      "sae_sel": 0.567
    }
  }
}
```

- `mse_pos/mse_whole`: Reconstruction error metrics
- `l0/l1`: Sparsity metrics
- `{probe_name}`: Delta selectivity results per probe

## Analysis

Exploratory notebooks in `explore/` provide:
- Result visualization and analysis
- Activation space exploration  
- Performance comparisons across models/layers
- Statistical significance testing

## Contributing

1. Follow existing code style and structure
2. Add new datasets via `utils.py` functions
3. Test changes with small model/dataset combinations first
4. Update documentation for new features

## Citation

If you use this code, please cite the Delta Selectivity paper [citation details to be added].
