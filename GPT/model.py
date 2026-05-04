import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import GPT2Tokenizer
from TransformerBlock import TransformerBlock
from pathlib import Path


class Onyx(nn.Module):
    """
    A transformer-based language model with causal self-attention.
    
    This model uses a stack of transformer blocks to process token sequences and
    generate predictions for the next token in the sequence. It supports token
    embeddings, positional embeddings, and causal masking to ensure that predictions
    at each position only depend on previous positions.
    """
    def __init__(self, vocab_size:int = 50_257, context_length: int = 1024, emb_size: int = 768, num_heads: int = 12, num_layers: int = 12, dropout: float = 0.1) -> None:
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
        
        # Tokenizer is loaded at model init so generation and training use a consistent encoding scheme
        self.encoder = GPT2Tokenizer.from_pretrained("gpt2")
        
        # Token embedding layer: maps token IDs to embedding vectors
        self.embedding = nn.Embedding(vocab_size, emb_size)
        
        # Positional embedding layer: adds positional information to token embeddings
        self.pos_embedding = nn.Embedding(context_length, emb_size)
        
        # Stack of transformer blocks for processing the embedded sequence
        self.transformer_layers = nn.ModuleList([TransformerBlock(emb_size, num_heads, 4*emb_size, dropout) for layer in range(num_layers)])
        
        self.ln_f = nn.LayerNorm(emb_size)
        
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
        # Explicit causal mask ensures stable token-level generation across all Transformer blocks
        # Upper triangle of -inf ensures masked positions have 0 attention weight after softmax
        causal_mask = torch.triu(torch.full((seq_length, seq_length), True, device=X.device), diagonal=1).bool()
        
        # Pass through each transformer block with the causal mask
        for layer in self.transformer_layers:
            X = layer(X, causal_mask)
        
        X = self.ln_f(X)
        # Project final embeddings to vocabulary logits for token prediction
        return self.linear(X)
    
    def generate(self, prompt: str, max_tokens: int = 100, top_p: float = 0.95, top_k: int = 50, temp: float = 0.7, repetition_penalty: float = 1.2) -> str:
        """
        Generate text continuation from a given prompt using combined top-p/top-k sampling.
        
        Uses both nucleus (top-p) and top-k sampling together with a repetition penalty
        to encourage diverse, coherent text generation while avoiding repetitive loops.
        
        Args:
            prompt (str): The input text prompt to continue from.
            max_tokens (int): Maximum number of tokens to generate. Defaults to 100.
            top_p (float): Cumulative probability threshold for nucleus sampling (0.0 to 1.0).
                          Defaults to 0.95.
            top_k (int): Keep only top k highest probability tokens. Defaults to 50.
            temp (float): Temperature for controlling randomness. Lower = more deterministic,
                         higher = more random. Defaults to 0.7.
            repetition_penalty (float): Penalty applied to tokens already in the sequence.
                                       >1.0 discourages repetition. Defaults to 1.2.
        
        Returns:
            str: The generated text, including the original prompt plus the continuation.
        """
        self.eval()
        # Tokenize the input prompt and convert to tensor with batch dimension
        encoded_prompt = self.encoder.encode(prompt)
        if len(encoded_prompt) == 0:
            raise ValueError("Prompt must contain at least one token. Please provide non-empty text input.")
        device = next(self.parameters()).device
        tokenized = torch.tensor(encoded_prompt, dtype=torch.long).unsqueeze(0).to(device)
        
        # Generate tokens one by one up to max_tokens
        for _ in range(max_tokens):
            # Take the last context_length tokens to maintain context window
            context = tokenized[:, -self.context_length:]
            
            # Get model predictions for the current context
            logits = self(context)
            
            # Sample the next token using combined top-p/top-k sampling with repetition penalty
            next_token = self._sample_top_p_top_k(logits[:, -1, :], tokenized, top_p, top_k, temp, repetition_penalty)
            
            # Append the new token to the sequence
            tokenized = torch.cat([tokenized, next_token], dim=1)
            
            # STOP if we hit newline or EOS
            if next_token.item() == self.encoder.eos_token_id:
                break
            if self.encoder.decode([next_token.item()]) == '\n':
                break
        
        # Decode the complete token sequence back to text
        return self.encoder.decode(tokenized[0].tolist())
    
    def _sample_top_p_top_k(self, logits: Tensor, token_history: Tensor, top_p: float = 0.9, top_k: int = 50, temperature: float = 1, repetition_penalty: float = 1.2) -> Tensor:
        """
        Sample from a distribution using combined top-p/top-k sampling with repetition penalty.
        
        This method applies:
        1. Repetition penalty to discourage repeated tokens
        2. Top-k filtering (keep only k highest probability tokens)
        3. Top-p filtering (nucleus sampling)
        4. Temperature scaling for randomness control
        
        Args:
            logits (Tensor): Shape [batch_size, vocab_size]. Raw model outputs.
            token_history (Tensor): Shape [batch_size, seq_len]. Previously generated tokens.
            top_p (float): Cumulative probability threshold (0.0 to 1.0). Default 0.9.
            top_k (int): Keep only top k highest probability tokens. Default 50.
            temperature (float): Temperature for controlling randomness. >1 = more random, <1 = more greedy.
            repetition_penalty (float): Penalty for repeated tokens (>1.0). Default 1.2.
        
        Returns:
            Tensor: Shape [batch_size, 1]. Sampled token IDs.
        """
        batch_size = logits.shape[0]
        
        # Apply repetition penalty: divide logits of tokens that appear in history
        if repetition_penalty != 1.0:
            for batch_idx in range(logits.shape[0]):
                unique_tokens = torch.unique(token_history[batch_idx])
                
                score = logits[batch_idx, unique_tokens]
                logits[batch_idx, unique_tokens] = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)

        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Convert to probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            kth_values = torch.topk(logits, top_k, dim=-1).values[:, -1, None]  # k-th largest
            logits = logits.masked_fill(logits < kth_values, float('-inf'))
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens where cumulative prob exceeds top_p (shift right to keep the token that crosses)
            sorted_to_remove = cumulative_probs - sorted_probs > top_p
            # Scatter back to original ordering
            to_remove = sorted_to_remove.scatter(1, sorted_indices, sorted_to_remove)
            logits = logits.masked_fill(to_remove, float('-inf'))
            
        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token