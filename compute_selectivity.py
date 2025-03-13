import argparse
import json
import os
import threading

import torch
import cupy as cp
from cuml.linear_model import LogisticRegression
from cuml.metrics import accuracy_score
from cuml.svm import LinearSVC, SVC
from cuml.model_selection import train_test_split

from utils import (
    get_models,
    get_datasets,
    get_layers,
    get_model_activations,
    get_sae_activations,
    get_binary_probes,
    get_dataset_name
)



def compute_selectivity(x, y):
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


    cp.random.shuffle(y_train)
    cp.random.shuffle(y_test)
    svc = LinearSVC(max_iter=1000, tol=1e-4)
    svc.fit(X_train, y_train)
    pred = svc.predict(X_test)
    control_acc = accuracy_score(y_test, pred)

    return float(task_acc - control_acc)


def compute_delta_selectivity(model_activations: List[torch.Tensor], # sample, pos, embedding
                              sae_activations: List[torch.Tensor], # sample, pos, embedding
                              targets: torch.Tensor): # sample,
    """
    model_activations, sae_activations: shape [N, Pos, D], where
       N = number of samples (called 'pos' in the description)
       Pos = Pos
       D = embedding dimension
    targets: shape [N, ] (binary labels)
    Returns: (model selectivity, sae selectivity)
    """
    model_activations = cp.from_dlpack(torch.utils.dlpack.to_dlpack(model_activations))
    sae_activations = cp.from_dlpack(torch.utils.dlpack.to_dlpack(sae_activations))
    targets = cp.from_dlpack(torch.utils.dlpack.to_dlpack(targets))

    model_selectivity = compute_selectivity(model_activations, target)
    sae_selectivity = compute_selectivity(sae_activations, target)

    return model_selectivity, sae_selectivity


def process_datasets_on_gpu(device, dataset_indices, output_dir):
    """
    Worker function (runs in a single thread) that:
      1) Sets the current CUDA device to gpu_id
      2) Loops over the given subset of dataset indices
      3) For each dataset index:
         - loops over get_models() and get_layers(model)
         - pulls the model_activations and sae_activations
         - loops over get_binary_probes(dataset_idx) => (name, pos, targets)
           ~ calls compute_delta_selectivity() on [pos]-th slice of the activations
         - writes out JSON results
    """

    models = get_models()
    #all_results = {}

    for dataset_idx in dataset_indices:
        #dataset_results = {}
        for model in models:
            #model_results = {}
            layers = get_layers(model)
            for layer in layers:
                
                model_acts = get_model_activations(model, layer, dataset_idx, device=device)
                repeats = torch.tensor([m.shape[0] for m in model_acts])
                sae_acts   = get_sae_activations(model, layer, dataset_idx, device=device)
                
                model_acts = cp.from_dlpack(torch.utils.dlpack.to_dlpack(torch.cat(model_acts, dim=0)))
                sae_acts = cp.from_dlpack(torch.utils.dlpack.to_dlpack(torch.cat(sae_acts, dim=0)))
                
                layer_results = {}
                for (probe_name, pos, targets) in get_binary_probes(dataset_idx):
                    targets = torch.repeat_interleave(targets, repeats)
                    sel_result = compute_delta_selectivity(
                        model_acts,
                        sae_acts,
                        targets
                    )

                    layer_results[probe_name] = sel_result
                # Write results to JSON.  Each thread writes a separate file.
                out_path = os.path.join(output_dir, f"{model}_{layer}L_{get_dataset_name(dataset_idx)}.json")
                with open(out_path, 'w') as f:
                    json.dump(all_results, f, indent=2)


                #model_results[layer] = layer_results
            #dataset_results[model] = model_results

        #all_results[dataset_idx] = dataset_results

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="Comma-separated list of GPU IDs, e.g. '0,1,2'")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Where to save JSON results per GPU")
    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]

    # Get the list of dataset indices
    dataset_indices = get_datasets()

    # Partition dataset_indices evenly among GPUs
    # e.g. if you have 3 GPUs and 8 datasets => each GPU handles a subset
    # GPU0: first 3, GPU1: next 3, GPU2: last 2
    n_gpus = len(gpu_ids)
    n_datasets = len(dataset_indices)
    splits = []
    start = 0
    for i in range(n_gpus):
        # slice up roughly equal subsets
        end = start + (n_datasets - start) // (n_gpus - i)
        splits.append(dataset_indices[start:end])
        start = end

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Spawn threads, one per GPU
    threads = []
    for i, gpu_id in enumerate(gpu_ids):
        subset = splits[i]
        t = threading.Thread(
            target=process_datasets_on_gpu,
            args=(gpu_id, subset, args.output_dir)
        )
        t.start()
        threads.append(t)

    # Wait for threads to finish
    for t in threads:
        t.join()

    print("All threads finished. JSON results written to", args.output_dir)


if __name__ == "__main__":
    main()
