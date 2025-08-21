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

The project evaluates on WikiData-derived binary classification tasks from [Marks & Tegmark (2023)](https://arxiv.org/pdf/2305.01610). Dataset download instructions are available in their [GitHub repository](https://github.com/wesg52/sparse-probing-paper). Each dataset consists of text documents mentioning names of popular people, with probes applied at the person's name positions.

**Data Directory Setup:**
The datasets should be placed in a `data/` directory with the following structure:
```
data/
├── wikidata_sorted_sex_or_gender.pyth.128.6000      # Gender classification
├── wikidata_sorted_political_party.pyth.128.3000   # Political affiliation  
├── wikidata_sorted_occupation_athlete.pyth.128.5000 # Athletic specialization
├── wikidata_sorted_is_alive.pyth.128.6000          # Life status
├── wikidata_sorted_occupation.pyth.128.6000        # General occupation
├── natural_lang_id.pyth.512.-1                     # Natural language identification
├── distribution_id.pyth.512.-1                     # Data distribution identification
└── ... (additional datasets)
```

Update the data paths in `utils.py` to match your local data directory structure.

| Feature | Dataset | Description |
|---------|---------|-------------|
| is_football | wikidata athlete | A dataset of text documents mentioning names of popular sports persons, probed at the names of those persons. The target class represents whether the person is a football player or not. |
| is_basketball | wikidata athlete | The target class represents whether the person is a basketball player or not. |
| is_baseball | wikidata athlete | The target class represents whether the person is a baseball player or not. |
| is_american_football | wikidata athlete | The target class represents whether the person is an American football player or not. |
| is_icehockey | wikidata athlete | The target class represents whether the person is an ice hockey player or not. |
| is_female | wikidata sex or gender | A dataset of text documents mentioning names of popular celebrities, probed at the names of those persons. The target class represents whether the person is a female (1) or male (0). |
| is_alive | wikidata is alive | A dataset of text documents mentioning names of popular celebrities, probed at the names of those persons. The target class represents whether the person is alive or not. |
| is_democratic | wikidata political party | A dataset of text documents mentioning names of popular political persons, probed at the names of those persons. The target class represents whether the person is a Democrat (1) or Republican (0). |
| is_singer | wikidata occupation | A dataset of text documents mentioning names of popular celebrities, probed at the names of those persons. The target class represents whether the person is a singer or not. |
| is_actor | wikidata occupation | The target class represents whether the person is an actor or not. |
| is_politician | wikidata occupation | The target class represents whether the person is a politician or not. |
| is_journalist | wikidata occupation | The target class represents whether the person is a journalist or not. |
| is_athlete | wikidata occupation | The target class represents whether the person is an athlete or not. |
| is_researcher | wikidata occupation | The target class represents whether the person is a researcher or not. |

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

Results are saved as JSON files in the specified `--output_directory` with the naming convention:
`{model_name}_{dataset_name}.json`

Example output file `EleutherAI_pythia-160m_wikidata_gender.json`:

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

**Metrics Explanation:**
- `mse_pos/mse_whole`: Reconstruction error metrics for SAE
- `l0/l1`: Sparsity metrics (L0 norm = number of active features, L1 norm = activation magnitude)
- `{probe_name}`: Delta selectivity results per binary classification probe
  - `model_sel`: Classification accuracy using original model activations
  - `sae_sel`: Classification accuracy using SAE-reconstructed activations
