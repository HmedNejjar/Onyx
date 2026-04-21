from pathlib import Path
from torch.utils.data import Dataset
from torch import Tensor
import numpy as np
import torch


class GPTDataset(Dataset):
    """
    PyTorch Dataset for GPT model training.
    
    Loads pre-tokenized text (stored as numpy arrays of token IDs) and creates
    (input, target) pairs by applying a sliding window. For each position i,
    the input is tokens[i:i+seq_length] and the target is tokens[i+1:i+seq_length+1],
    enabling next-token prediction training.
    
    Attributes:
        tokens (np.ndarray): 1D array of token IDs loaded from file.
        seq_length (int): Length of input sequences. Defaults to 1024.
    """
    def __init__(self, tokens_file: str | Path, seq_length: int = 1024) -> None:
        """
        Initialize the GPT dataset.
        
        Args:
            tokens_file (str | Path): Path to the numpy file containing pre-tokenized text
                                     as a 1D array of token IDs (dtype: int32 or int64).
            seq_length (int): Length of each sequence in tokens. Controls the context window
                             for next-token prediction. Defaults to 1024.
        """
        super().__init__()
        self.tokens = np.load(tokens_file)
        self.seq_length = seq_length
        
    def __len__(self):
        """
        Return the number of available samples in the dataset.
        
        Returns:
            int: Number of valid (input, target) pairs that can be created.
                 Equals total tokens minus sequence length to avoid going past the end.
        """
        return len(self.tokens) - self.seq_length
    
    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        """
        Retrieve a single (input, target) pair at the given index.
        
        Creates a next-token prediction pair by sliding a window:
        - Input (x):  tokens at positions [idx, idx+seq_length)
        - Target (y): tokens at positions [idx+1, idx+seq_length+1)
        
        This ensures that each position in the input predicts the next token.
        
        Args:
            idx (int): Index of the sample to retrieve. Must be in range [0, len(dataset)).
        
        Returns:
            tuple[Tensor, Tensor]: (input_ids, target_ids) as PyTorch long tensors.
                                  Both have shape (seq_length,).
        """
        # Input sequence: current position to current + seq_length
        x = torch.tensor(self.tokens[idx:idx+self.seq_length], dtype=torch.long)
        # Target sequence: next token to next + seq_length (shifted by 1)
        y = torch.tensor(self.tokens[idx+1:idx+self.seq_length+1], dtype=torch.long)
        
        return x, y
    
if __name__ == "__main__":
    # Example usage: instantiate dataset and verify shapes
    dataset = GPTDataset("G:\\Projects\\Python\\Onyx\\Data\\wikitext_train.npy")
    print(f"Dataset length: {len(dataset)}")
    
    # Test retrieval of first sample
    x, y = dataset[0]
    print(f"Input shape: {x.shape}, Target shape: {y.shape}")