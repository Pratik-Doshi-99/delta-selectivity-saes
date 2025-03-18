import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk
from transformer_lens import HookedTransformer
from utils import get_data_path, preprocess_data, get_binary_probes
import json
from pythia_sae import SAE
import re
import logging
import sys

# Set up logging to stdout with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

#########################################################
#                   Dataset / Collate                  #
#########################################################

class CustomDataset(Dataset):
    def __init__(self, dataset, dataset_idx, offset=0):
        """
        Initialize with the original dataset and dataset index.
        offset is used to keep track of the 'true' dataset index if we skip some rows.
        """
        self.dataset = dataset
        self.dataset_idx = dataset_idx
        self.offset = offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        tokenized_prompt, positions = preprocess_data(sample, self.dataset_idx)
        binary_probes = get_binary_probes(sample, self.dataset_idx)
        # Return the tokenized prompt, positions, and the "global" index (offset + idx)
        return tokenized_prompt, positions, (self.offset + idx), binary_probes


def collate_fn(batch):
    """
    Pad tokenized prompts and prepare batch data.
    Args:
        batch: List of (tokenized_prompt, positions, global_idx) tuples.
    Returns:
        input_ids: Padded tensor [batch_size, max_seq_len].
        attention_mask: Tensor indicating non-padded positions [batch_size, max_seq_len].
        positions_list: List of position lists for each sample.
        global_indices: Dataset-wide sample indices corresponding to each batch row.
    """
    tokenized_prompts, positions_list, global_indices, binary_probes = zip(*batch)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        tokenized_prompts, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return input_ids, attention_mask, positions_list, global_indices, binary_probes


#########################################################
#      Default Processor: Write Activations to Disk     #
#########################################################

def write_activations_and_metrics_to_disk(
    model_acts_accum,
    sae_acts_accum,
    reconstruction_metrics_accum,
    probes,
    layers,
    shard_num,
    model_name_safe,
    sae_name_safe,
    dataset_dir,
    output_size
):
    """
    Write the accumulated activations and reconstruction metrics to disk.
    This function expects the same data structures used in the main loop.
    """
    logger.info(f"Saving tensors to file for shard {shard_num}...")

    model_acts_dir = dataset_dir / 'model-activations'
    sae_acts_dir = dataset_dir / 'sae-activations'
    model_acts_dir.mkdir(parents=True, exist_ok=True)
    sae_acts_dir.mkdir(parents=True, exist_ok=True)

    # Save model & SAE activations
    for layer in layers:
        if len(model_acts_accum[layer]) == 0:
            continue  # Skip empty accumulations

        model_shard_path = model_acts_dir / f"{model_name_safe}_{layer}L_{shard_num}.pt"
        sae_shard_path = sae_acts_dir / f"{sae_name_safe}_{layer}L_{shard_num}.pt"

        torch.save(model_acts_accum[layer], model_shard_path)
        torch.save(sae_acts_accum[layer], sae_shard_path)
        logger.info(f"Saved {model_shard_path} and {sae_shard_path} to disk")

        # Reset after saving
        model_acts_accum[layer].clear()
        sae_acts_accum[layer].clear()

    # Save metrics shard if it is non-empty
    if reconstruction_metrics_accum:
        shard_metrics_path = dataset_dir / f"reconstruction_metrics_{shard_num}.json"
        with open(shard_metrics_path, 'w') as f:
            json.dump(reconstruction_metrics_accum, f, indent=4)
        logger.info(f"Saved {shard_metrics_path} to disk")
        reconstruction_metrics_accum.clear()


#########################################################
#    Alternative Processor: Put Activations in a Queue  #
#########################################################

def enqueue_activations_and_metrics(
    model_acts_accum,
    sae_acts_accum,
    metrics_accum,
    probes,
    layers,
    shard_num,
    model_name_safe,
    sae_name_safe,
    dataset_dir,
    output_size,
    queue
):
    """
    Example alternative to writing data to disk: push it into a queue for further processing.
    Note: 'queue' can be any data structure or multiprocessing queue, etc.
    """
    logger.info(f"Enqueuing activations for shard {shard_num}...")
    #logger.info(f"Length of probes enqueued: {len(probes)}")
    # For demonstration, we put everything in the queue as one object
    queue.put({
        "shard_num": shard_num,
        "model_acts": {layer: data for layer, data in model_acts_accum.items()},
        "sae_acts": {layer: data for layer, data in sae_acts_accum.items()},
        "metrics": {layer: data for layer, data in metrics_accum.items()},  # shallow copy
        "probes": probes[:]
    })
    # Clear the original accumulations
    #for layer in layers:
    #    model_acts_accum[layer].clear()
    #    sae_acts_accum[layer].clear()
    #    metrics_accum[layer].clear()
    #probes.clear()


#########################################################
#          Main Logic in a Callable Function            #
#########################################################

def compute_activations_and_metrics(args, activation_processor=None, processor_kwargs=None):
    """
    Computes model and SAE activations and processes them (defaults to writing to disk).

    :param args: Namespace of arguments (from argparse).
    :param activation_processor: Function that receives the accumulations
        when it's time to 'flush' them (write to disk, queue them, etc.).
    :param processor_kwargs: Additional kwargs passed into activation_processor.
    """

    if activation_processor is None:
        # Default: write to disk
        activation_processor = write_activations_and_metrics_to_disk

    if processor_kwargs is None:
        processor_kwargs = {}

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

    # Slice dataset
    start = args.start_index
    if args.n_samples is not None:
        end = min(start + args.n_samples, len(dataset))
    else:
        end = len(dataset)
    dataset = dataset.select(range(start, end))

    # Create custom dataset
    custom_dataset = CustomDataset(dataset, args.dataset_idx, offset=start)

    # Load model and tokenizer
    global tokenizer  # needed for collate_fn
    logger.info("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.encode(tokenizer.eos_token)[0]
    model = HookedTransformer.from_pretrained(args.model_name).to(device)

    # Load SAEs
    logger.info("Loading SAE...")
    sae_handle = SAE(device=device, sae_layer_template=args.sae_layer_template)
    sae_handle.load_many(args.sae_name, layers)

    # Prepare accumulators
    model_acts_accum = {layer: [] for layer in layers} # num_layers * num_samples
    sae_acts_accum = {layer: [] for layer in layers} #num_layers * num_samples
    metrics = {layer: [] for layer in layers} #num_layers
    probes = []
    sample_count = 0

    # Prepare output directory
    dataset_dir = Path(args.output_directory)

    # Create DataLoader
    dataloader = DataLoader(
        custom_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False  # Keep order for consistent sharding
    )
    logger.info("Starting Inference...")

    # Process dataset
    for batch_idx, (input_ids, attention_mask, positions_list, global_indices, binary_probes) in enumerate(dataloader):
        logger.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}...")
        #logger.info(f"Length of batched binary_probes={len(binary_probes)}, probes={len(probes)}")
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
            #logger.info(f"Forward pass completed for batch {batch_idx + 1}.")

            # Prepare per-sample metric storage for this batch
            batch_metrics = [{} for _ in range(batch_size)]
            for i in range(batch_size):
                batch_metrics[i]['sample_idx'] = int(global_indices[i])


            probes += [binary_probes[i] if p is not None and len(p) > 0 else None for i,p in enumerate(positions_list)]
            # Process each layer
            for layer in layers:
                hook_name = re.sub("<layer>", str(layer), args.model_hook_template)
                full_hidden_state_batch = cache[hook_name]  # [batch_size, seq_len, hidden_dim]

                # SAE activations
                decoded_batch, sae_acts_full_batch = sae_handle.compute_activations(full_hidden_state_batch, layer=layer)
                #logger.info(f"SAE activations computed for layer {layer} in batch {batch_idx + 1}.")

                # Iterate through samples
                for sample_i in range(batch_size):
                    positions = positions_list[sample_i]
                    if positions is not None and len(positions) > 0:
                        original_sample = full_hidden_state_batch[sample_i, positions, :]
                        decoded_sample = decoded_batch[sample_i, positions, :]
                        sae_acts_sample = sae_acts_full_batch[sample_i, positions, :]

                        # Compute reconstruction metrics
                        mse_pos = ((original_sample - decoded_sample) ** 2).mean(dim=-1).sum().cpu().item()
                        mse_whole = ((full_hidden_state_batch[sample_i, :, :] - decoded_batch[sample_i, : , :]) ** 2).mean(dim=-1).sum().cpu().item()
                        #l0 proportion
                        l0 = (sae_acts_sample != 0).sum(dim=-1).sum().cpu().item()
                        l1 = sae_acts_sample.abs().sum(dim=-1).sum().cpu().item()

                        # # Store metrics by token position
                        # for pos_idx, pos in enumerate(positions):
                        #     pos_str = f"pos_{pos}"
                        #     if pos_str not in batch_metrics[sample_i]:
                        #         batch_metrics[sample_i][pos_str] = {}
                        #     batch_metrics[sample_i][pos_str][f"layer_{layer}"] = {
                        #         'mse': mse[pos_idx],
                        #         'l0': l0[pos_idx],
                        #         'l1': l1[pos_idx]
                        #     }

                        # Accumulate activations
                        metrics[layer].append({
                            'idx':sample_i,
                            'mse_pos':mse_pos,
                            'mse_whole':mse_whole,
                            'count_whole':full_hidden_state_batch.shape[1],
                            'count_pos':len(positions),
                            'l0': l0,
                            'l1': l1
                            })
                        model_acts_accum[layer].append(original_sample.cpu())
                        sae_acts_accum[layer].append(sae_acts_sample.to_sparse().cpu())
                    else:
                        # For samples that have no positions, append None
                        model_acts_accum[layer].append(None)
                        sae_acts_accum[layer].append(None)
                        metrics[layer].append({
                                'idx':sample_i,
                                'mse_pos':None,
                                'mse_whole':None,
                                'count_whole':None,
                                'count_pos':None,
                                'l0': None,
                                'l1': None
                            })


        # Clear cache to free memory
        del cache
        sample_count += batch_size

        # Flush to activation_processor if we have reached the shard size
        if sample_count % args.output_size == 0:
            shard_num = sample_count // args.output_size
            activation_processor(
                model_acts_accum,
                sae_acts_accum,
                metrics,
                probes,
                layers,
                shard_num,
                model_name_safe,
                sae_name_safe,
                dataset_dir,
                args.output_size,
                **processor_kwargs
            )
            model_acts_accum = {layer: [] for layer in layers} # num_layers * num_samples
            sae_acts_accum = {layer: [] for layer in layers} #num_layers * num_samples
            metrics = {layer: [] for layer in layers} #num_layers
            probes = []

    # If there are any leftover activations or metrics, flush them
    # (e.g. if the total samples wasn't a multiple of output_size).
    leftover_data = any(len(model_acts_accum[layer]) > 0 for layer in layers) or len(metrics) > 0
    if leftover_data:
        shard_num = (sample_count // args.output_size) + 1
        activation_processor(
            model_acts_accum,
            sae_acts_accum,
            metrics,
            probes,
            layers,
            shard_num,
            model_name_safe,
            sae_name_safe,
            dataset_dir,
            args.output_size,
            **processor_kwargs
        )
        model_acts_accum = {layer: [] for layer in layers} # num_layers * num_samples
        sae_acts_accum = {layer: [] for layer in layers} #num_layers * num_samples
        metrics = {layer: [] for layer in layers} #num_layers
        probes = []


#########################################################
#                 Command-Line Entry                    #
#########################################################

def main():
    parser = argparse.ArgumentParser(description="Compute and process model/SAE activations with reconstruction metrics.")
    parser.add_argument('--model_name', type=str, required=True,
                        help="Hugging Face model name (e.g., 'meta-llama/Meta-Llama-3-8B')")
    parser.add_argument('--sae_name', type=str, required=True,
                        help="Identifier for SAE models (e.g., 'EleutherAI/sae-llama-3-8b-32x')")
    parser.add_argument('--dataset_idx', type=int, required=True,
                        help="Dataset index for get_data_path")
    parser.add_argument('--start_index', type=int, default=0,
                        help="Start from this dataset index for metrics computation")
    parser.add_argument('--n_samples', type=int, default=None,
                        help="Number of samples to use (optional)")
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
                        help='E.g. "blocks.<layer>.hook_resid_post"')
    parser.add_argument('--sae_layer_template', type=str, default='layers.<layer>',
                        help='E.g. "layers.<layer>"')
    args = parser.parse_args()

    # By default, we will just write to disk. But we can swap in any other function.
    compute_activations_and_metrics(args)


if __name__ == '__main__':
    main()
