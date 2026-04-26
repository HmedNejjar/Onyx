import sys
sys.path.insert(3, "G:\\Projects\\Python\\Onyx")

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from TransformerBlock import TransformerBlock
from Tokenizer.Encoder import BPE
from pathlib import Path


class Onyx(nn.Module):
    """
    A transformer-based language model with causal self-attention.
    
    This model uses a stack of transformer blocks to process token sequences and
    generate predictions for the next token in the sequence. It supports token
    embeddings, positional embeddings, and causal masking to ensure that predictions
    at each position only depend on previous positions.
    """
    def __init__(self, vocab_size:int = 30_000, context_length: int = 1024, emb_size: int = 1024, num_heads: int = 16, num_layers: int = 12, dropout: float = 0.1, tokenizer: str | Path = Path(r'G:\Projects\Python\Onyx\Tokenizer\BPE_30k.json')) -> None:
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
        
        self.encoder = BPE(vocab_size=self.vocab_size)
        self.encoder.load(tokenizer)
        
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
    
    def generate(self, prompt: str, max_tokens: int = 100, top_p: float = 0.9, temp: float = 0.1) -> str:
        """
        Generate text continuation from a given prompt using autoregressive sampling.
        
        >>> Get input prompt, then iteratively generates new tokens
        by sampling from the model's probability distribution. It uses top-p (nucleus)
        sampling with temperature scaling to control the randomness of generation.
        
        Args:
            prompt (str): The input text prompt to continue from.
            max_tokens (int): Maximum number of tokens to generate. Defaults to 100.
            top_p (float): Cumulative probability threshold for nucleus sampling (0.0 to 1.0).
                          Higher values allow more diverse outputs. Defaults to 0.9.
            temp (float): Temperature for controlling randomness. Lower values make output
                         more deterministic, higher values more random. Defaults to 0.1.
        
        Returns:
            str: The generated text, including the original prompt plus the continuation.
        """
        # Tokenize the input prompt and convert to tensor with batch dimension
        encoded_prompt = self.encoder.encode(prompt)
        if len(encoded_prompt) == 0:
            raise ValueError("Prompt must contain at least one token. Please provide non-empty text input.")
        tokenized = torch.tensor(encoded_prompt, dtype=torch.long).unsqueeze(0)
        
        # Generate tokens one by one up to max_tokens
        for _ in range(max_tokens):
            # Take the last context_length tokens to maintain context window
            context = tokenized[:, -self.context_length:]
            
            # Get model predictions for the current context
            logits = self(context)
            
            # Sample the next token using top-p sampling with temperature
            next_token = self._sample_top_p(logits[:, -1, :], top_p, temp)
            
            # Append the new token to the sequence
            tokenized = torch.cat([tokenized, next_token], dim=1)
        
        # Decode the complete token sequence back to text
        return self.encoder.decode(tokenized[0].tolist())
            
    def _sample_top_p(self, logits:Tensor, top_p:float = 0.9, temperature:float = 0.8):
        """
        Sample from a distribution using top-p (nucleus) sampling with temperature scaling.
        
        Args:
            logits (Tensor): Shape [batch_size, vocab_size]. Raw model outputs.
            top_p (float): Cumulative probability threshold (0.0 to 1.0). Default 0.9.
            temperature (float): Temperature for controlling randomness. >1 = more random, <1 = more greedy.
        
        Returns:
            Tensor: Shape [batch_size, 1]. Sampled token IDs.
        """
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Convert to probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Compute cumulative sum
        cum_sum = torch.cumsum(sorted_probs, dim=-1)
        
        # Find the cutoff: tokens where cumsum <= top_p
        sorted_indices_to_remove = cum_sum > top_p
        
        # Always keep at least the top token (avoid removing everything)
        sorted_indices_to_remove[:, 0] = False
        
        # Map back to original indices and zero out low-probability tokens
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[torch.arange(probs.shape[0]).unsqueeze(1), indices_to_remove] = 0.0
        
        # Renormalize
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Sample
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
            
                
                