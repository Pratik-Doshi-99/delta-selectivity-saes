import argparse
import json
import logging
import sys
import torch
from pathlib import Path
from queue import Queue
import threading
import os
import random

# Assume these come from your codebase:
from utils import (
    get_models,
    get_datasets,
    get_layers,
    get_binary_probes,
    get_dataset_name
)
# These come from your reorganized code module:
#   compute_activations_and_metrics
#   enqueue_activations_and_metrics
from activations import compute_activations_and_metrics, enqueue_activations_and_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)



def compute_selectivity(x, y, device='cpu'):
    """
    x_1d: shape [N, ] (single dimension's activations for all N samples)
    y: shape [N, ]
    We'll fit a logistic regression, measure accuracy, then shuffle y and measure again.
    """

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    svc = LinearSVC(max_iter=1000, tol=1e-4)
    svc.fit(X_train, y_train)
    pred = svc.predict(X_test)
    task_acc = accuracy_score(y_test, pred)

    if 'cuda' in device:
        cp.random.shuffle(y_train)
        cp.random.shuffle(y_test)
    else:
        random.shuffle(y_train)
        random.shuffle(y_test)
    svc = LinearSVC(max_iter=1000, tol=1e-4)
    svc.fit(X_train, y_train)
    pred = svc.predict(X_test)
    control_acc = accuracy_score(y_test, pred)

    return float(task_acc - control_acc)


def compute_delta_selectivity(model_activations: List[torch.Tensor], # sample * pos, embedding
                              sae_activations: List[torch.Tensor], # sample * pos, embedding
                              targets: torch.Tensor, # sample, 1
                              device: 'cpu'):
    """
    model_activations, sae_activations: shape [N, Pos, D], where
       N = number of samples (called 'pos' in the description)
       Pos = Pos
       D = embedding dimension
    targets: shape [N, ] (binary labels)
    device: device on which the selectivity must be computed
    Returns: (model selectivity, sae selectivity)
    """
    if 'cuda' in device:
        model_activations = cp.from_dlpack(torch.utils.dlpack.to_dlpack(model_activations))
        sae_activations = cp.from_dlpack(torch.utils.dlpack.to_dlpack(sae_activations))
        targets = cp.from_dlpack(torch.utils.dlpack.to_dlpack(targets))

    model_selectivity = compute_selectivity(model_activations, targets)
    sae_selectivity = compute_selectivity(sae_activations, targets)

    return model_selectivity, sae_selectivity





def activation_consumer_thread(
    queue_in,
    dataset_idx,
    second_device,
    output_dir,
    model
):
    """
    Worker thread function that:
      - Dequeues shard data from `enqueue_activations_and_metrics`.
      - Moves them to `second_device`.
      - Accumulates them for the entire dataset.
      - Once a sentinel `None` is received, calls get_binary_probes and
        runs compute_delta_selectivity on each probe, saving to disk.
    """
    if 'cuda' in second_device:
        import cupy as cp
        from cuml.metrics import accuracy_score
        from cuml.svm import LinearSVC, SVC
        from cuml.model_selection import train_test_split
    else:
        from sklearn.svm import LinearSVC, SVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
    
    all_model_acts = {}
    all_sae_acts = {}
    all_metrics = {}
    all_probes = []

    while True:
        item = queue_in.get()
        if item is None:
            # Sentinel: means the entire dataset has been processed
            logger.info(f"[Consumer] Got sentinel for dataset {dataset_idx}. Finishing up.")
            break

        shard_num = item["shard_num"]
        model_acts_dict = item["model_acts"]   # { layer -> list of Tensors (num_samples) or None }
        sae_acts_dict   = item["sae_acts"]     # { layer -> list of Tensors (num_samples) or None }
        metrics = item['metrics']              # { layer -> list of Tensors (num_samples) or None }
        probes = item['probes']                # [
                                               #    [(probe1, [], 1), (probe2, [], 0), ... (probe_n, [], 1)]
                                               #    ...
                                               #    [(probe1, [], 1), (probe2, [], 0), ... (probe_n, [], 1)]
                                               # ]
        

        # aggregate the output sizes to make the entire dataset for probing
        for layer, acts_list in model_acts_dict.items():
            if layer not in all_model_acts:
                all_model_acts[layer] = []
            all_model_acts[layer] += [a.to(second_device) for a in acts_list if a is not None]

        for layer, acts_list in sae_acts_dict.items():
            if layer not in all_sae_acts:
                all_sae_acts[layer] = []
            all_sae_acts[layer] += [a.to(second_device) for a in acts_list if a is not None]
        
        for layer, metric in metrics:
            if layer not in all_metrics:
                all_metrics[layer] = []
            all_metrics[layer] += [m for m in metric if m is not None]
            
        all_probes += probes
        queue_in.task_done()

    # -------------------------------------------------------------------------
    # Entire dataset is accumulated beyond this point
    
    #computing layer-wise metrics
    for layer in all_metrics.keys():
        sum_mse_pos = 0
        sum_mse_whole = 0
        count_pos = 0
        count_whole = 0
        l0 = 0
        l1 = 0
        for m in all_metrics[layer]:
            sum_mse_pos += m['mse_pos']
            sum_mse_whole += m['mse_whole']
            count_pos += m['count_pos']
            count_whole += m['count_whole']
            l0 += m['l0']
            l1 += m['l1']
        all_metrics[layer] = {
            'mse_pos': sum_mse_pos / count_pos,
            'mse_whole': sum_mse_whole / count_whole,
            'l0': l0 / count_pos, 
            'l1': l1 / count_pos
        }


    target_tensor = torch.tensor([[p[2] for p in sample_i] for sample_i in all_probes]) # num_samples, n_probes
    probe_names = [p[0] for p in all_probes[0]] # string of probe names, corresponding to the n_probes in target_tensor
    
    logger.info(f"[Consumer] Received {len(probe_names)} binary probes for dataset {dataset_idx}.")
    logger.info(f"[Consumer] Probes Received for dataset {dataset_idx}: {probe_names}")
    # For each probe, we do something like:
    #   targets = <some shape [num_samples]>
    #   model_activations = <list of length num_samples> shape [pos, dim]
    #   sae_activations   = <list of length num_samples> shape [pos, dim]
    
    for layer in all_model_acts.keys():
                
        model_acts = all_model_acts[layer] # list of tensors for that layer. Each tensor: seq, embed
        repeats = torch.tensor([m.shape[0] for m in model_acts])
        sae_acts   = all_sae_acts[layer] # list of tensors for that layer. Each tensor: seq, embed
        
        model_acts = torch.cat(model_acts, dim=0)
        sae_acts = torch.cat(sae_acts, dim=0)

        if 'cuda' in second_device:
            # model acts and sae_Acts modified: [n_samples, seq, embed] -> [n_samples * seq, embed]
            model_acts = cp.from_dlpack(torch.utils.dlpack.to_dlpack(model_acts))
            sae_acts = cp.from_dlpack(torch.utils.dlpack.to_dlpack(sae_acts))
        
        for i in range(len(probe_names)):
            targets = torch.repeat_interleave(target_tensor[:,i], repeats)



            model_sel, sae_sel = compute_delta_selectivity(
                model_acts,
                sae_acts,
                targets
            )

            all_metrics[layer][probe_names[i]] = {'model_sel':model_sel, 'sae_sel':sae_sel, }
        # Write results to JSON.  Each thread writes a separate file.
    
    out_path = os.path.join(output_dir, f"{model}_{get_dataset_name(dataset_idx)}.json")
    with open(out_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"[Consumer] Wrote delta selectivity results to {out_path} for dataset {dataset_idx}.")
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device1", help="Main device for model forward passes")
    parser.add_argument("--device2", help="Second device for metric computations")
    parser.add_argument("--output_directory", required=True, help="Where to save final results")
    parser.add_argument('--model_name', type=str, required=True,
                        help="Hugging Face model name (e.g., 'meta-llama/Meta-Llama-3-8B')")
    parser.add_argument('--sae_name', type=str, required=True,
                        help="Identifier for SAE models (e.g., 'EleutherAI/sae-llama-3-8b-32x')")
    parser.add_argument('--start_index', type=int, default=0,
                        help="Start from this dataset index for metrics computation")
    parser.add_argument('--n_samples', type=int, default=None,
                        help="Number of samples to use (optional)")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for processing")
    parser.add_argument('--layers', type=str, required=True,
                        help="Comma-separated list of layer indices (e.g., '0,1,2')")
    parser.add_argument('--output_size', type=int, default=1024,
                        help="Number of samples per shard file")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of workers for DataLoader")
    parser.add_argument('--model_hook_template', type=str, default='blocks.<layer>.hook_resid_post',
                        help='E.g. "blocks.<layer>.hook_resid_post"')
    parser.add_argument('--sae_layer_template', type=str, default='layers.<layer>',
                        help='E.g. "layers.<layer>"')
    # ... plus any other arguments you need for compute_activations_and_metrics ...
    args = parser.parse_args()

    if args.device is None:
        device1 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.device2 is None:
        device2 = 'cuda:1' if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else 'cpu'

    device1 = args.device
    device2 = args.device2
    output_directory = args.output_directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Example: you could loop over each dataset in get_datasets() or
    # each model in get_models(). We'll just do datasets as requested:
    datasets = get_datasets()  # e.g. [0, 1, 2,...]
    logger.info(f"Will process dataset indices: {datasets}")

    for dataset_idx in datasets:
        logger.info(f"Starting dataset {dataset_idx}...")

        # Prepare a queue
        q = Queue()

        # Spawn consumer thread
        consumer_args = (q, dataset_idx, device2, output_directory)
        consumer_thread = threading.Thread(target=activation_consumer_thread, args=consumer_args)
        consumer_thread.start()

        # Now call compute_activations_and_metrics with the queue-based processor
        # For demonstration, we show minimal arguments. 
        # You would fill in the real ones (model name, layers, etc.)
        from argparse import Namespace
        dummy_args = Namespace(
            model_name=args.model_name,
            sae_name=args.sae_name,
            dataset_idx=dataset_idx,
            start_index=0,
            n_samples=None,
            device=device1,
            batch_size=args.batch_size,
            layers=args.layers,
            output_size=args.output_size,
            output_directory=output_directory,
            num_workers=args.num_workers,
            model_hook_template=args.model_hook_template,
            sae_layer_template=args.sae_layer_template
        )

        compute_activations_and_metrics(
            dummy_args,
            activation_processor=enqueue_activations_and_metrics,
            processor_kwargs={"queue": q}
        )

        # Once done, send sentinel to let the consumer thread finalize
        q.put(None)
        # Wait for the consumer thread to finish
        consumer_thread.join()

        logger.info(f"Dataset {dataset_idx} is fully processed.\n")

    logger.info("All datasets processed. Exiting.")


if __name__ == "__main__":
    main()
