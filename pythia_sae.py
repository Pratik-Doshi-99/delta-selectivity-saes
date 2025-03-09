import torch
import torch.nn as nn
from typing import List
from sparsify.sparsify import Sae
import einops
import re
from sparse_tensor import SparseTensor

class SAE:
    def __init__(self, device, sae_layer_template='layers.<layer>'):
        # Initialize encoder and decoder as empty modules.
        self.encoder = None
        self.decoder = None
        self.device = device
        self.sae_layer_template = sae_layer_template
        self.sae_layers = {}

    def load_many(self, sae_name:str, layers: List[int]):
        print('Loading SAEs')
        saes = Sae.load_many(sae_name, device=self.device)
        print('Finished Loading SAEs')
        sae_keys = [(layer, re.sub("<layer>",str(layer),self.sae_layer_template)) for layer in layers]
        for key in sae_keys:
            if key[1] not in saes:
                raise ValueError(f"SAE for layer {key[1]} not found in {sae_name}")
            self.sae_layers[key[0]] = saes[key[1]].to(self.device)
            print(f'Layer {key[0]} laoded')
        


    # can create a function to give compressed activations later on
    def compute_activations(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """
        Pass the input through the encoder to obtain the latent representation.
        
        Args:
            x (torch.Tensor): Tensor of (batch, seq, embedding)
            
        Returns:
            torch.Tensor: Tensor of (batch, seq, embedding)
        """
        batch = x.shape[0]
        sae = self.sae_layers[layer]
        x_mod = einops.rearrange(x, "b s e -> (b s) e") # collapsing to (batch * sequence, embedding)
        with torch.inference_mode():
            fwd = sae.forward(x_mod)
            sae_latents = torch.zeros(fwd.latent_acts.shape[0], sae.cfg.num_latents, device=fwd.latent_acts.device, dtype=fwd.latent_acts.dtype)
            sae_latents.scatter_(1, fwd.latent_indices, fwd.latent_acts)
            sae_latents = einops.rearrange(sae_latents, "(b s) e -> b s e", b = batch)
            #latent_indices = einops.rearrange(fwd.latent_indices, "(b s) e -> b s e", b = batch)
            #latent_acts = einops.rearrange(fwd.latent_acts, "(b s) e -> b s e", b = batch)
            #latent_indices = einops.rearrange(fwd.latent_indices, "(b s) e -> b s e", b = batch)
            #st = SparseTensor(
            #    latent_acts,
            #    latent_indices,
            #    (latent_acts.shape[0], latent_acts.shape[1], sae.cfg.num_latents),
            #    2
            #)
            y_og = einops.rearrange(fwd.sae_out, "(b s) e -> b s e", b = batch)
            
        return y_og, sae_latents
            
            
        
