from torch import Tensor, nn

class TransformerBlock(nn.Module):
    """
    A single Transformer Decoder Block with self-attention and feed-forward network.
        
    Architecture Flow:
        1. Input X
        2. └─→ LayerNorm → Multi-Head Self-Attention → Dropout → Residual Add (+) → X1
        3. └─→ LayerNorm → Feed-Forward Network → Dropout → Residual Add (+) → X2
        4. Output X2
    
    Attributes:
        attn (nn.MultiheadAttention): Multi-head self-attention mechanism
        ff (nn.Sequential): Position-wise feed-forward network (2-layer MLP)
        norm1 (nn.LayerNorm): Layer normalization before attention sub-layer
        norm2 (nn.LayerNorm): Layer normalization before feed-forward sub-layer
        dropout (nn.Dropout): Dropout layer applied to residual connections
    """
    def __init__(self, emb_size: int, num_heads: int, ff_dim: int, dropout: float = 0.1) -> None:
        """
        Initialize the Transformer Block.
        
        Args:
            emb_size (int): Embedding dimension (d_model). The size of token representations
                           throughout the block. Must be divisible by num_heads.
            num_heads (int): Number of attention heads. The embedding dimension will be
                            split into num_heads equal parts. Typically 8, 12, or 16.
                            emb_size % num_heads must equal 0.
            ff_dim (int): Hidden dimension of the feed-forward network. Typically set to
                         4 * emb_size (e.g., 3072 for emb_size=768). The FFN expands to
                         this dimension, applies activation, then projects back to emb_size.
            dropout (float): Dropout probability. Applied to attention weights, FFN outputs,
                            and residual connections. Range: [0.0, 1.0). Default: 0.1.
        
        Raises:
            AssertionError: If emb_size is not divisible by num_heads (checked by PyTorch)
        """
        super().__init__()
        
        # Self-attention mechanism that allows each token to attend to all other tokens
        self.attn = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Position-wise feed-forward network applied to each token independently
        self.ff = nn.Sequential(nn.Linear(emb_size, ff_dim), nn.ReLU(), nn.Linear(ff_dim, emb_size), nn.Dropout(dropout))
        
        # LayerNorm normalizes across the feature dimension (emb_size) for each token
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        
        # Applied to the output of each sub-layer before adding to the residual path, in forward()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass through the Transformer Block.
               
        Args:
            X (Tensor): Input tensor of shape [batch_size, seq_length, emb_size]
                       Contains the token representations from previous layer.
            mask (Tensor): Attention mask of shape [seq_length, seq_length] or 
                          [batch_size, seq_length, seq_length]. For causal (autoregressive)
                          models, this should be an upper triangular matrix with -inf
                          above the diagonal to prevent attending to future tokens.
        
        Returns:
            Tensor: Output tensor of same shape as input [batch_size, seq_length, emb_size]
                   Contains the transformed token representations after attention
                   and feed-forward processing.
        """
        # Self-Attention with normalization
        attn_output, _ = self.attn(X, X, X, attn_mask=mask)
        X = self.norm1(X + self.dropout(attn_output))
        
        # Feed-Forward Network with normalization
        ff_output = self.ff(X)
        X = self.norm2(X + self.dropout(ff_output))
        
        return X
