import torch
from typing import Tuple

class SparseTensor:
    def __init__(self, act_tensor=None, idx_tensor=None, uncomp_shape=None, uncomp_dim=None):
        # Sparse representation: indices and corresponding activations.
        self.index_tensor = idx_tensor         # Stores indices for nonzero elements.
        self.activation_tensor = act_tensor    # Stores the nonzero values.
        self.uncompressed_dim_idx = uncomp_dim # The dimension along which compression was done.
        self.uncompressed_shape = uncomp_shape   # The full shape of the original tensor.

    @staticmethod
    def compress(dense_tensor: torch.Tensor, compression_dim: int) -> "SparseTensor":
        pass

    def decompress(self) -> torch.Tensor:
        dense_tensor = torch.zeros(self.uncompressed_shape, device=self.activation_tensor.device, dtype=self.activation_tensor.dtype)
        print('Dense Tensor Created:',dense_tensor.shape)
        print(self.uncompressed_dim_idx)
        print(self.index_tensor)
        print(self.activation_tensor)
        dense_tensor.scatter_(self.uncompressed_dim_idx, self.index_tensor, self.activation_tensor)
        return dense_tensor

    def serialize(self) -> dict:
        """
        Serializes the SparseTensor into a dictionary.
        This state dictionary can be saved using torch.save.
        """
        state = {
            "index_tensor": self.index_tensor,
            "activation_tensor": self.activation_tensor,
            "uncompressed_dim_idx": self.uncompressed_dim_idx,
            "uncompressed_shape": self.uncompressed_shape,
        }
        return state

    @staticmethod
    def deserialize(state: dict) -> "SparseTensor":
        """
        Reconstructs a SparseTensor from a state dictionary (e.g., one loaded via torch.load).
        
        Args:
            state (dict): A state dictionary containing the sparse attributes.
            
        Returns:
            SparseTensor: The reconstructed SparseTensor instance.
        """
        st = SparseTensor()
        st.index_tensor = state["index_tensor"]
        st.activation_tensor = state["activation_tensor"]
        st.uncompressed_dim_idx = state["uncompressed_dim_idx"]
        st.uncompressed_shape = state["uncompressed_shape"]
        return st
