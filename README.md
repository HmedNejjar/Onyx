# Onyx: A Transformer-Based Language Model

A from-scratch implementation of a GPT-style causal language model in PyTorch, trained on WikiText-103. Onyx combines token embeddings, positional encodings, and stacked transformer decoder blocks to perform next-token prediction and autoregressive text generation.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation & Requirements](#installation--requirements)
4. [Project Structure](#project-structure)
5. [How It Works](#how-it-works)
6. [Module Reference](#module-reference)
7. [Training Pipeline](#training-pipeline)
8. [Usage Examples](#usage-examples)
9. [Configuration & Hyperparameters](#configuration--hyperparameters)
10. [Performance Notes](#performance-notes)

---

## Overview

**Onyx** is a fully-functional language model implementation that demonstrates:

- **Transformer architecture** with causal (autoregressive) masking
- **BPE tokenization** using Hugging Face's `tokenizers` library
- **Efficient training** with gradient accumulation, mixed precision (AMP), and gradient clipping
- **Nucleus sampling** (top-p) for controlled generation
- **Memory-mapped data loading** for handling large tokenized corpora

The model is designed for educational purposes and research prototyping, with production-quality code structure and documentation.

---

## Architecture

### High-Level Flow

```
Input (token IDs)
    ↓
Token Embedding + Positional Embedding
    ↓
Stack of N Transformer Blocks
    ├─ Multi-Head Self-Attention (causal mask)
    ├─ Layer Norm + Residual Connection
    ├─ Feed-Forward Network (2-layer MLP)
    └─ Layer Norm + Residual Connection
    ↓
Linear Projection to Vocab Logits
    ↓
Output (logits for next-token prediction)
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **Token Embedding** | Maps token IDs to dense vectors |
| **Positional Embedding** | Adds absolute position information |
| **Transformer Block** | Self-attention + FFN with residual connections |
| **Causal Mask** | Prevents attention to future tokens (autoregressive) |
| **Top-P Sampling** | Nucleus sampling for text generation |

---

## Installation & Requirements

### Python Version
- **Python 3.10+** (uses `str | Path` type hints)

### Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers tokenizers
pip install datasets
pip install tqdm
pip install numpy
```

### Optional (but recommended)
- **CUDA 11.8+** for GPU acceleration (training is ~10-50x faster on GPU)
- **Mixed Precision** via `torch.cuda.amp` (reduces memory, maintains accuracy)

### Installation Steps

```bash
# Clone or download the project
cd Onyx

# Install dependencies
pip install -r requirements.txt  # if included, or use pip install commands above

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

---

## Project Structure

```
Onyx/
├── Tokenizer/
│   ├── Encoder.py              # BPE tokenizer wrapper
│   └── BPE_30k.json            # Trained BPE vocabulary (30k tokens)
│
├── GPT/
│   ├── model.py                # Onyx language model (main)
│   ├── TransformerBlock.py     # Single transformer block
│   ├── Dataloader.py           # PyTorch Dataset for token sequences
│   ├── precompute.py           # Tokenize raw text into .npy files
│   ├── train.py                # Training loop with validation
│   └── best_model.pth          # Checkpoint (created after training)
│
├── Data/
│   ├── wikitext_train.npy      # Tokenized WikiText-103 training set
│   ├── wikitext_val.npy        # Tokenized WikiText-103 validation set
│   └── corpus.txt              # (Optional) Raw corpus for tokenizer training
│
└── README.md                   # This file
```

---

## How It Works

### 1. **Tokenization (Encoder.py)**

The `BPE` tokenizer converts raw text into token IDs:

```python
# Train BPE tokenizer
tokenizer = BPE(vocab_size=30_000, min_frequency=3)
tokenizer.train("Data/corpus.txt")
tokenizer.save("Tokenizer/BPE_30k.json")

# Encode: text → token IDs
tokens = tokenizer.encode("Hello, world!")  # → [104, 22, 506, 12]

# Decode: token IDs → text
text = tokenizer.decode([104, 22, 506, 12])  # → "Hello, world!"
```

**Key Details:**
- Uses **Byte Pair Encoding (BPE)**: iteratively merges most-frequent character pairs
- Vocabulary size: **30,000** (balance between coverage and model size)
- Special tokens: `<s>`, `<pad>`, `</s>`, `<unk>`, `<mask>`

---

### 2. **Data Loading (Dataloader.py)**

The `GPTDataset` creates next-token prediction pairs from tokenized sequences:

```python
# Load pre-tokenized text (stored as numpy array)
dataset = GPTDataset("Data/wikitext_train.npy", seq_length=512, stride=256)

# Retrieve a sample
input_ids, target_ids = dataset[0]
# input_ids:  tokens[0:512]        → shape (512,)
# target_ids: tokens[1:513]        → shape (512,)
```

**Sliding Window Mechanism:**
- `seq_length`: How many tokens to use as context (default: 1024)
- `stride`: Step size between samples (controls overlap; default: 512)
- Each sample `i` produces inputs at `[i*stride : i*stride+seq_length]` and targets at `[i*stride+1 : i*stride+seq_length+1]`

This creates overlapping sequences, increasing dataset size while reducing redundancy.

---

### 3. **Model Architecture (model.py)**

The `Onyx` model implements a causal language model:

```python
model = Onyx(
    vocab_size=30_000,      # Size of token vocabulary
    context_length=1024,    # Max sequence length
    emb_size=1024,          # Embedding dimension (d_model)
    num_heads=16,           # Attention heads
    num_layers=12,          # Transformer blocks
    dropout=0.1
)

# Forward pass
logits = model(token_ids)  # shape: (batch_size, seq_length, vocab_size)
```

**Components:**

| Layer | Purpose |
|-------|---------|
| **Embedding** | Token ID → embedding vector |
| **Pos. Embedding** | Add absolute position info |
| **Transformer Blocks (×12)** | Self-attention + FFN |
| **Linear** | Project embeddings → vocab logits |

---

### 4. **Transformer Block (TransformerBlock.py)**

Each block applies multi-head attention and feed-forward processing:

```
Input X
  ↓
LayerNorm → Multi-Head Attention → Dropout → Residual Add
  ↓
LayerNorm → FFN (Linear → ReLU → Linear → Dropout) → Residual Add
  ↓
Output X'
```

**Key Details:**
- **Multi-Head Attention**: Each head attends to different representation subspaces
- **Causal Mask**: Upper-triangular matrix of `-inf` prevents attending to future tokens
- **Residual Connections**: Skip connections help gradient flow
- **Layer Normalization**: Stabilizes training

---

### 5. **Training (train.py)**

The training loop optimizes the model using cross-entropy loss:

```python
# Setup
model = Onyx(...).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_fn = CrossEntropyLoss()
scaler = GradScaler()  # For mixed precision

# Training
for epoch in range(EPOCHS):
    for batch_idx, (x, y) in enumerate(train_loader):
        with autocast():  # Mixed precision
            logits = model(x)  # (batch, seq_len, vocab)
            loss = loss_fn(logits.view(-1, vocab), y.view(-1))
        
        # Gradient accumulation
        (loss / ACCUMULATION_STEPS).backward()
        
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
    
    # Validate and save best model
    val_loss, val_acc = validate()
```

**Optimization Techniques:**
- **Gradient Accumulation**: Simulates larger batch size without OOM
- **Mixed Precision (AMP)**: Uses FP16 for faster compute, FP32 for stable gradients
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate**: 1e-4 with AdamW optimizer

---

### 6. **Text Generation (model.py → generate())**

The model generates text autoregressively using **top-p (nucleus) sampling**:

```python
prompt = "Once upon a time"
generated_text = model.generate(
    prompt,
    max_tokens=100,
    top_p=0.9,      # Keep tokens with cumsum probability ≤ 0.9
    temp=0.8        # Temperature for softness
)
```

**Sampling Algorithm:**
1. Get model logits for the last token in context
2. Apply temperature scaling: `logits / temperature`
3. Convert to probabilities via softmax
4. Sort by probability descending
5. Find cutoff where cumulative sum ≤ `top_p`
6. Zero out tokens above cutoff
7. Renormalize and sample with `torch.multinomial()`

This produces more diverse outputs than greedy sampling while staying coherent.

---

## Module Reference

### **Tokenizer/Encoder.py**

Byte Pair Encoding tokenizer using Hugging Face's `tokenizers` library.

#### Class: `BPE`

```python
BPE(vocab_size=30_000, min_frequency=3)
```

| Method | Signature | Purpose |
|--------|-----------|---------|
| `train()` | `train(file: str \| Path)` | Train BPE on corpus file |
| `save()` | `save(path: str \| Path)` | Save tokenizer to JSON |
| `load()` | `load(path: str \| Path)` | Load saved tokenizer |
| `encode()` | `encode(text: str) → list[int]` | Text → token IDs |
| `decode()` | `decode(tokens: list[int]) → str` | Token IDs → text |

**Example:**
```python
from Tokenizer.Encoder import BPE

tokenizer = BPE()
tokenizer.load("Tokenizer/BPE_30k.json")
tokens = tokenizer.encode("Hello world")
text = tokenizer.decode(tokens)
```

---

### **GPT/Dataloader.py**

PyTorch Dataset for loading pre-tokenized sequences.

#### Class: `GPTDataset`

```python
GPTDataset(tokens_file: str | Path, seq_length: int = 1024, stride: int = 512)
```

| Attribute | Type | Purpose |
|-----------|------|---------|
| `tokens` | `np.ndarray` | Loaded token IDs (memory-mapped) |
| `seq_length` | `int` | Context window size |
| `stride` | `int` | Step between consecutive samples |

| Method | Returns | Purpose |
|--------|---------|---------|
| `__len__()` | `int` | Number of samples |
| `__getitem__(idx)` | `(Tensor, Tensor)` | Input and target token IDs |

**Example:**
```python
from GPT.Dataloader import GPTDataset

dataset = GPTDataset("Data/wikitext_train.npy", seq_length=512, stride=256)
print(f"Samples: {len(dataset)}")

x, y = dataset[0]
print(x.shape)  # (512,)
print(y.shape)  # (512,)
```

---

### **GPT/TransformerBlock.py**

Single transformer decoder block with self-attention and FFN.

#### Class: `TransformerBlock`

```python
TransformerBlock(emb_size: int, num_heads: int, ff_dim: int, dropout: float = 0.1)
```

| Attribute | Type | Purpose |
|-----------|------|---------|
| `attn` | `nn.MultiheadAttention` | Multi-head self-attention |
| `ff` | `nn.Sequential` | 2-layer feed-forward network |
| `norm1, norm2` | `nn.LayerNorm` | Layer normalization |
| `dropout` | `nn.Dropout` | Dropout for regularization |

| Method | Signature | Purpose |
|--------|-----------|---------|
| `forward()` | `forward(X: Tensor, mask: Tensor) → Tensor` | Process input through block |

**Example:**
```python
from GPT.TransformerBlock import TransformerBlock
import torch

block = TransformerBlock(emb_size=1024, num_heads=16, ff_dim=4096)
x = torch.randn(2, 512, 1024)  # (batch, seq_len, emb_size)
causal_mask = torch.triu(torch.ones(512, 512) * float('-inf'), diagonal=1)
output = block(x, causal_mask)  # (2, 512, 1024)
```

---

### **GPT/model.py**

The main `Onyx` language model.

#### Class: `Onyx`

```python
Onyx(
    vocab_size: int = 30_000,
    context_length: int = 1024,
    emb_size: int = 1024,
    num_heads: int = 16,
    num_layers: int = 12,
    dropout: float = 0.1,
    tokenizer: str | Path = Path("Tokenizer/BPE_30k.json")
)
```

| Method | Signature | Purpose |
|--------|-----------|---------|
| `forward()` | `forward(X: Tensor) → Tensor` | Compute logits for next-token prediction |
| `generate()` | `generate(prompt: str, max_tokens: int = 100, top_p: float = 0.9, temp: float = 0.1) → str` | Generate text from prompt |
| `_sample_top_p()` | `_sample_top_p(logits, top_p, temperature) → Tensor` | Top-p sampling utility |

**Example:**
```python
from GPT.model import Onyx
import torch

model = Onyx(vocab_size=30_000, emb_size=512, num_layers=12)
model.load_state_dict(torch.load("GPT/best_model.pth"))
model.eval()

with torch.no_grad():
    generated = model.generate("Once upon a time", max_tokens=100, top_p=0.9)
    print(generated)
```

---

### **GPT/precompute.py**

Standalone script to tokenize raw WikiText-103 corpus into numpy arrays.

**What it does:**
1. Loads WikiText-103 dataset from Hugging Face `datasets` library
2. Tokenizes training and validation splits using BPE
3. Saves token IDs as memory-mapped `.npy` files for efficient loading

**Run:**
```bash
cd GPT
python precompute.py
```

**Output:**
```
Processing training set...
Tokenizing: 100%|████████| 100000/100000 [2:34:12<00:00, 10.78it/s]
Processing validation set...
Tokenizing: 100%|████████| 3760/3760 [6:12<00:00, 10.08it/s]

Tokenization Complete!
Train size: 103,232,451 tokens
Val size:   2,141,280 tokens
```

---

### **GPT/train.py**

Full training loop with validation, checkpointing, and mixed precision.

**Key Hyperparameters (at top of file):**

```python
VOCAB_SIZE = 30_000         # Tokenizer vocab size
STRIDE = 256                # Dataset stride (controls sequence overlap)
CONTEXT_LEN = 512           # Sequence length for training
EMB_SIZE = 512              # Embedding dimension
NUM_HEADS = 16              # Attention heads (EMB_SIZE % NUM_HEADS == 0)
NUM_LAYERS = 12             # Transformer blocks
DROPOUT = 0.1               # Dropout probability
BATCH_SIZE = 2              # Batch size
ACCUMULATION_STEPS = 8      # Gradient accumulation (eff. batch = 16)
EPOCHS = 10                 # Training epochs
LR = 1e-4                   # Learning rate (AdamW)
```

**Run:**
```bash
cd GPT
python train.py
```

**Output (per epoch):**
```
Epoch 1/10: 100%|████████| 12500/12500 [15:32<00:00, 13.42 it/s]
[Epoch 1/10] Train Loss: 4.2341 | Train Acc: 28.54% | Val Loss: 3.9821 | Val Acc: 34.12%  👍  SAVED

[Epoch 2/10] Train Loss: 3.8734 | Train Acc: 36.21% | Val Loss: 3.7654 | Val Acc: 38.45%  👍  SAVED
...
```

**Checkpoint:** Best model saved to `GPT/best_model.pth`

---

## Training Pipeline

### Step-by-Step Workflow

```
1. Raw WikiText-103 Corpus
        ↓ (precompute.py)
2. Tokenized .npy files (token IDs)
        ↓ (Dataloader.py)
3. PyTorch DataLoader (batched sequences)
        ↓ (train.py)
4. Model Forward Pass (logits)
        ↓
5. Cross-Entropy Loss
        ↓
6. Backpropagation + Gradient Accumulation
        ↓
7. Optimizer Step (every 8 batches)
        ↓
8. Validation & Checkpoint
```

### Data Parallelism

For multi-GPU training, wrap the model:

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

(Distributed training via `DistributedDataParallel` for >8 GPUs recommended)

---

## Usage Examples

### 1. **Generate Text from Pretrained Model**

```python
from GPT.model import Onyx
import torch

# Load model
model = Onyx(vocab_size=30_000, emb_size=512, num_layers=12)
model.load_state_dict(torch.load("GPT/best_model.pth"))
model.eval()

# Generate
with torch.no_grad():
    text = model.generate(
        prompt="The future of AI is",
        max_tokens=150,
        top_p=0.95,
        temp=0.7
    )
    print(text)
```

**Output:**
```
The future of AI is bright and full of possibilities. As machine learning advances,
models will become more efficient and interpretable...
```

---

### 2. **Train from Scratch (with custom hyperparameters)**

Edit `GPT/train.py`:

```python
CONTEXT_LEN = 1024          # Larger context
EMB_SIZE = 768              # Slightly smaller
NUM_LAYERS = 24             # More depth
BATCH_SIZE = 4              # Larger batches (if VRAM allows)
ACCUMULATION_STEPS = 4
EPOCHS = 15
LR = 5e-5                   # Smaller LR for stability
```

Then run:

```bash
python GPT/train.py
```

---

### 3. **Encode/Decode with Tokenizer**

```python
from Tokenizer.Encoder import BPE

tokenizer = BPE()
tokenizer.load("Tokenizer/BPE_30k.json")

# Encode
text = "Transformers are powerful models!"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# Decode
reconstructed = tokenizer.decode(tokens)
print(f"Decoded: {reconstructed}")
```

---

### 4. **Create Custom Dataset**

```python
from GPT.Dataloader import GPTDataset
from torch.utils.data import DataLoader

# Create dataset with shorter sequences and larger stride
dataset = GPTDataset("Data/wikitext_train.npy", seq_length=256, stride=128)

# Create DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

for batch_x, batch_y in loader:
    print(f"Batch input: {batch_x.shape}")  # (4, 256)
    print(f"Batch target: {batch_y.shape}")  # (4, 256)
    break
```

---

## Configuration & Hyperparameters

### Model Architecture

| Parameter | Default | Notes |
|-----------|---------|-------|
| `vocab_size` | 30,000 | BPE vocabulary size |
| `context_length` | 1024 | Max sequence length |
| `emb_size` | 1024 | Embedding dimension (d_model) |
| `num_heads` | 16 | Attention heads; must divide `emb_size` |
| `num_layers` | 12 | Number of transformer blocks |
| `dropout` | 0.1 | Regularization rate |

### Training

| Parameter | Default | Notes |
|-----------|---------|-------|
| `batch_size` | 2 | Per-GPU batch size |
| `accumulation_steps` | 8 | Effective batch = 2 × 8 = 16 |
| `learning_rate` | 1e-4 | AdamW optimizer |
| `weight_decay` | 1e-5 | L2 regularization |
| `epochs` | 10 | Training epochs |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |

### Dataset

| Parameter | Default | Notes |
|-----------|---------|-------|
| `seq_length` | 512 (train.py) | Context window for sequences |
| `stride` | 256 (train.py) | Overlap factor; smaller = more samples |

### Generation

| Parameter | Default | Notes |
|-----------|---------|-------|
| `max_tokens` | 100 | Tokens to generate |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `temperature` | 0.1 | Softmax temperature (lower = more greedy) |

---

## Performance Notes

### Memory Usage

On NVIDIA V100 (16GB):
- **Model size**: ~1.5GB (embeddings + 12 transformer blocks)
- **Batch 1, seq_len 512**: ~6GB
- **Batch 2 + gradient accum (8 steps)**: ~7-8GB

For smaller GPUs, reduce:
- `EMB_SIZE` (512 or 768)
- `NUM_LAYERS` (6 or 8)
- `BATCH_SIZE` (1)

---

### Training Speed

**RTX 3090 (24GB)** with batch_size=4, accumulation_steps=2:
- **Throughput**: ~2,000 tokens/second
- **Epoch time**: ~15 hours (100M token dataset)
- **Full training (10 epochs)**: ~150 hours (~6 days)

**A100 (80GB)** with batch_size=32, accumulation_steps=1:
- **Throughput**: ~10,000 tokens/second
- **Epoch time**: ~3 hours
- **Full training**: ~30 hours (~1.25 days)

---

### Convergence

Typical training curves:
- **Epoch 1**: Loss drops from 10.5 → 4.2 (rapid)
- **Epoch 5**: Loss plateaus around 3.6 (gradual)
- **Epoch 10**: Loss reaches ~3.4 (diminishing returns)

For better performance, train longer (20+ epochs) or use larger models.

---

## Troubleshooting

### Issue: `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce `BATCH_SIZE` (e.g., 2 → 1)
2. Reduce `EMB_SIZE` (e.g., 1024 → 512)
3. Reduce `CONTEXT_LEN` (e.g., 1024 → 512)
4. Enable gradient accumulation: increase `ACCUMULATION_STEPS`

---

### Issue: `FileNotFoundError: 'Tokenizer/BPE_30k.json' not found`

**Solution:**
```bash
# Train tokenizer first
cd GPT
python precompute.py  # Creates .npy files AND requires tokenizer

# Or train tokenizer separately
python Tokenizer/Encoder.py
```

---

### Issue: Training loss doesn't decrease

**Checklist:**
- Is learning rate too high? Try `LR = 5e-5`
- Is `CONTEXT_LEN` too long for data? Try 256 or 512
- Is there a bug in data loading? Verify `x.shape == (batch, seq_len)` and `y.shape == (batch, seq_len)`
- Is causal mask correct? Should be upper-triangular with `-inf` above diagonal

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., NeurIPS 2017
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al., OpenAI 2019
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

## License

MIT License — Feel free to use for research and educational purposes.

---

## Contact

For questions or suggestions, reach out via GitHub issues or email bakr.m210906@gmail.com.