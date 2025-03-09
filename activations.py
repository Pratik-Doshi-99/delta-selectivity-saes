import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk
from transformer_lens import HookedTransformer
from utils import get_data_path, preprocess_data
import json
from pythia_sae import SAE
import re
import logging
import sys

# Set up logging to stdout with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, dataset, dataset_idx, offset=0):
        """
        Initialize with the original dataset and dataset index.
        offset is used to keep track of the "true" dataset index if we skip some rows.
        """
        self.dataset = dataset
        self.dataset_idx = dataset_idx
        self.offset = offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        tokenized_prompt, positions = preprocess_data(sample, self.dataset_idx)
        # Return the tokenized prompt, positions, and the "global" index (offset + idx)
        return tokenized_prompt, positions, (self.offset + idx)

def collate_fn(batch):
    """
    Pad tokenized prompts and prepare batch data.
    Args:
        batch: List of (tokenized_prompt, positions, global_idx) tuples from CustomDataset.
    Returns:
        input_ids: Padded tensor [batch_size, max_seq_len].
        attention_mask: Tensor indicating non-padded positions [batch_size, max_seq_len].
        positions_list: List of position lists for each sample.
        global_indices: The dataset-wide sample indices corresponding to each batch row.
    """
    tokenized_prompts, positions_list, global_indices = zip(*batch)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        tokenized_prompts, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return input_ids, attention_mask, positions_list, global_indices

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compute and save model and SAE activations with reconstruction metrics.")
    parser.add_argument('--model_name', type=str, required=True,
                        help="Hugging Face model name (e.g., 'meta-llama/Meta-Llama-3-8B')")
    parser.add_argument('--sae_name', type=str, required=True,
                        help="Identifier for SAE models (e.g., 'EleutherAI/sae-llama-3-8b-32x')")
    parser.add_argument('--dataset_idx', type=int, required=True,
                        help="Dataset index for get_data_path")
    parser.add_argument('--start_index', type=int, default=0,
                        help="Start from this dataset index for metrics computation")
    parser.add_argument('--n_samples', type=int, default=None,
                        help="Number of samples to use (optional). If provided, only that many samples are processed.")
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="Device to run computation (e.g., 'cuda:0')")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for processing")
    parser.add_argument('--layers', type=str, required=True,
                        help="Comma-separated list of layer indices (e.g., '0,1,2')")
    parser.add_argument('--output_size', type=int, default=1024,
                        help="Number of samples per shard file")
    parser.add_argument('--output_directory', type=str, required=True,
                        help="Output directory for saving activations and metrics")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of workers for DataLoader")
    parser.add_argument('--model_hook_template', type=str, default='blocks.<layer>.hook_resid_post',
                        help='E.g. "blocks.<layer>.hook_resid_post". The layer numbers replace the <layer> placeholder.')
    parser.add_argument('--sae_layer_template', type=str, default='layers.<layer>',
                        help='E.g. "layers.<layer>". The layer numbers replace the <layer> placeholder.')
    args = parser.parse_args()

    logger.info("Validating Arguments...")
    assert args.output_size % args.batch_size == 0, \
        "Output size must be a multiple of batch size as activations are saved with batch granularity"

    logger.info("Computing activations with the following arguments:")
    logger.info(args)

    # Process arguments
    model_name_safe = args.model_name.replace('/', '_')
    sae_name_safe = args.sae_name.replace('/', '_')

    layers = [int(l) for l in args.layers.split(',')]
    logger.info("Configured Model Hook Points:")
    logger.info([re.sub("<layer>", str(l), args.model_hook_template) for l in layers])
    logger.info("Configured SAEs:")
    logger.info([re.sub("<layer>", str(l), args.sae_layer_template) for l in layers])

    device = args.device
    logger.info("Loading Dataset...")
    # Load dataset
    dataset_path = get_data_path(args.dataset_idx)
    dataset = load_from_disk(dataset_path)

    # Slice dataset based on start_index and n_samples (no random shuffle)
    start = args.start_index
    if args.n_samples is not None:
        end = min(start + args.n_samples, len(dataset))
    else:
        end = len(dataset)
    dataset = dataset.select(range(start, end))

    # Create custom dataset
    custom_dataset = CustomDataset(dataset, args.dataset_idx, offset=start)

    # Load model and tokenizer
    global tokenizer  # Make tokenizer accessible in collate_fn

    logger.info("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.encode(tokenizer.eos_token)[0]
    model = HookedTransformer.from_pretrained(args.model_name).to(device)

    # Load SAEs
    logger.info("Loading SAE...")
    sae_handle = SAE(device=device, sae_layer_template=args.sae_layer_template)
    sae_handle.load_many(args.sae_name, layers)

    # Set up output directories
    dataset_dir = Path(args.output_directory)
    model_acts_dir = dataset_dir / 'model-activations'
    sae_acts_dir = dataset_dir / 'sae-activations'
    model_acts_dir.mkdir(parents=True, exist_ok=True)
    sae_acts_dir.mkdir(parents=True, exist_ok=True)

    # Initialize accumulation lists
    model_acts_accum = {layer: [] for layer in layers}
    sae_acts_accum = {layer: [] for layer in layers}
    reconstruction_metrics_accum = []
    sample_count = 0

    # Create DataLoader
    dataloader = DataLoader(
        custom_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False  # Keep order for consistent sharding
    )
    logger.info("Starting Inference...")

    # Process dataset using DataLoader
    for batch_idx, (input_ids, attention_mask, positions_list, global_indices) in enumerate(dataloader):
        logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}...")
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        batch_size = input_ids.size(0)

        # Cache activations
        cache = {}
        def caching_hook(activation, hook):
            cache[hook.name] = activation

        hook_names = [re.sub("<layer>", str(layer), args.model_hook_template) for layer in layers]
        hooks = [(hook_name, caching_hook) for hook_name in hook_names]

        with torch.inference_mode():
            # Forward pass
            _ = model.run_with_hooks(input_ids, attention_mask=attention_mask, fwd_hooks=hooks)
            logger.info(f"Forward pass completed for batch {batch_idx + 1}.")

            # Prepare per-sample metric storage for this batch
            batch_metrics = [{} for _ in range(batch_size)]

            # Fill in sample_idx (which is the "global" dataset index)
            for i in range(batch_size):
                batch_metrics[i]['sample_idx'] = int(global_indices[i])

            # Process each layer
            for layer in layers:
                hook_name = re.sub("<layer>", str(layer), args.model_hook_template)
                full_hidden_state_batch = cache[hook_name]  # [batch_size, seq_len, hidden_dim]

                # SAE activations
                decoded_batch, sae_acts_full_batch = sae_handle.compute_activations(full_hidden_state_batch, layer=layer)
                logger.info(f"SAE activations computed for layer {layer} in batch {batch_idx + 1}.")

                # Iterate through batch samples
                for sample_i in range(batch_size):
                    positions = positions_list[sample_i]
                    if positions is not None and len(positions) > 0:
                        original_sample = full_hidden_state_batch[sample_i, positions, :]  # [num_positions, hidden_dim]
                        #print(decoded_batch.shape, sample_i, positions)
                        decoded_sample = decoded_batch[sample_i, positions, :]             # [num_positions, hidden_dim]
                        sae_acts_sample = sae_acts_full_batch[sample_i, positions, :]     # [num_positions, latent_dim]
    
                        # Compute reconstruction metrics
                        mse = ((original_sample - decoded_sample) ** 2).mean(dim=-1).cpu().tolist()
                        l0 = (sae_acts_sample != 0).sum(dim=-1).cpu().tolist()
                        l1 = sae_acts_sample.abs().sum(dim=-1).cpu().tolist()
    
                        # Store metrics by token position
                        for pos_idx, pos in enumerate(positions):
                            pos_str = f"pos_{pos}"
                            if pos_str not in batch_metrics[sample_i]:
                                batch_metrics[sample_i][pos_str] = {}
                            batch_metrics[sample_i][pos_str][f"layer_{layer}"] = {
                                'mse': mse[pos_idx],
                                'l0': l0[pos_idx],
                                'l1': l1[pos_idx]
                            }
    
                        # Accumulate activations
                        model_acts_accum[layer].append(original_sample.cpu())
                        sae_acts_accum[layer].append(sae_acts_sample.to_sparse().cpu())
                        #logger.info(f"Metrics computed for sample {sample_i+1}/{batch_size} batch {batch_idx + 1} layer {layer}.")
                    else:
                        #add dummy
                        model_acts_accum[layer].append(None)
                        sae_acts_accum[layer].append(None)

            # Append batch metrics
            reconstruction_metrics_accum.extend(batch_metrics)
            

        del cache
        sample_count += batch_size

        # Save activations & metrics in shards
        if sample_count % args.output_size == 0:
            logger.info(f"Saving tensors to file.")
            shard_num = sample_count // args.output_size

            # Save model & SAE activations
            for layer in layers:
                model_shard_path = model_acts_dir / f"{model_name_safe}_{layer}L_{shard_num}.pt"
                sae_shard_path = sae_acts_dir / f"{sae_name_safe}_{layer}L_{shard_num}.pt"
                torch.save(model_acts_accum[layer], model_shard_path)
                torch.save(sae_acts_accum[layer], sae_shard_path)
                logger.info(f"Saved {model_shard_path} and {sae_shard_path} to disk")
                model_acts_accum[layer] = []
                sae_acts_accum[layer] = []

            # Save metrics shard
            shard_metrics_path = dataset_dir / f"reconstruction_metrics_{shard_num}.json"
            with open(shard_metrics_path, 'w') as f:
                json.dump(reconstruction_metrics_accum, f, indent=4)
            logger.info(f"Saved {shard_metrics_path} to disk")
            reconstruction_metrics_accum = []

    # Save any remaining activations
    if model_acts_accum[layers[0]]:
        shard_num = (sample_count // args.output_size) + 1
        for layer in layers:
            model_shard_path = model_acts_dir / f"{model_name_safe}_{layer}L_{shard_num}.pt"
            sae_shard_path = sae_acts_dir / f"{sae_name_safe}_{layer}L_{shard_num}.pt"
            torch.save(model_acts_accum[layer], model_shard_path)
            torch.save(sae_acts_accum[layer], sae_shard_path)
            logger.info(f"Saved {model_shard_path} and {sae_shard_path} to disk")

        # Save leftover metrics shard
        shard_metrics_path = dataset_dir / f"reconstruction_metrics_{shard_num}.json"
        with open(shard_metrics_path, 'w') as f:
            json.dump(reconstruction_metrics_accum, f, indent=4)
        logger.info(f"Saved {shard_metrics_path} to disk")
    else:
        # If no leftover activations, but leftover metrics (unlikely but safe to handle)
        if reconstruction_metrics_accum:
            shard_num = (sample_count // args.output_size) + 1
            shard_metrics_path = dataset_dir / f"reconstruction_metrics_{shard_num}.json"
            with open(shard_metrics_path, 'w') as f:
                json.dump(reconstruction_metrics_accum, f, indent=4)
            logger.info(f"Saved {shard_metrics_path} to disk")

if __name__ == '__main__':
    main()
