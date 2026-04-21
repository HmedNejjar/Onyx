import torch
from torch import Tensor, nn
from TransformerBlock import TransformerBlock


class Onyx(nn.Module):
    """
    A transformer-based language model with causal self-attention.
    
    This model uses a stack of transformer blocks to process token sequences and
    generate predictions for the next token in the sequence. It supports token
    embeddings, positional embeddings, and causal masking to ensure that predictions
    at each position only depend on previous positions.
    """
    def __init__(self, vocab_size:int = 30_000, context_length: int = 1024, emb_size: int = 1024, num_heads: int = 16, num_layers: int = 12, dropout: float = 0.1) -> None:
        """
        Initialize the Onyx language model.
        
        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens). Defaults to 30,000.
            context_length (int): Maximum sequence length the model can process. Defaults to 1024.
            emb_size (int): Dimensionality of token and positional embeddings. Defaults to 1024.
            num_heads (int): Number of attention heads in each transformer block. Defaults to 16.
            num_layers (int): Number of transformer blocks in the model. Defaults to 12.
            dropout (float): Dropout probability applied throughout the model. Defaults to 0.1.
        """
        super().__init__()
        
        # Store model configuration parameters
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Token embedding layer: maps token IDs to embedding vectors
        self.embedding = nn.Embedding(vocab_size, emb_size)
        
        # Positional embedding layer: adds positional information to token embeddings
        self.pos_embedding = nn.Embedding(context_length, emb_size)
        
        # Stack of transformer blocks for processing the embedded sequence
        self.transformer_layers = nn.ModuleList([TransformerBlock(emb_size, num_heads, 4*emb_size, dropout) for layer in range(num_layers)])
        
        # Output linear layer: projects embedding dimension back to vocabulary size for token prediction
        self.linear = nn.Linear(emb_size, vocab_size)
    
    def forward(self, X: Tensor) -> Tensor:
        """
        Forward pass of the Onyx language model.
        
        Args:
            X (Tensor): Input tensor of shape (batch_size, seq_length) containing token IDs.
        
        Returns:
            Tensor: Output logits of shape (batch_size, seq_length, vocab_size) for next-token prediction.
        """
        # Extract batch and sequence dimensions from input
        batch_size, seq_length = X.size()
        
        # Create position indices for the sequence and expand to batch dimension
        positions = torch.arange(0, seq_length, device=X.device).unsqueeze(0).expand(batch_size, -1)
        
        # Combine token embeddings with positional embeddings
        X = self.embedding(X) + self.pos_embedding(positions)
        
        # Create causal mask: prevent attention to future tokens (for autoregressive generation)
        # Upper triangle of -inf ensures masked positions have 0 attention weight after softmax
        causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=X.device) * float('-inf'), diagonal=1)
        
        # Pass through each transformer block with the causal mask
        for layer in self.transformer_layers:
            X = layer(X, causal_mask)
        
        # Project final embeddings to vocabulary logits for token prediction
        return self.linear(X)