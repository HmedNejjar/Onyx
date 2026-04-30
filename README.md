# Onyx: A Transformer-Based Language Model

A from-scratch implementation of a GPT-style causal language model in PyTorch. Combines token embeddings, positional encodings, and stacked transformer decoder blocks for next-token prediction and autoregressive text generation.

---

## What Is This?

**Onyx** is a complete transformer language model implementation with three main components:

1. **Model Architecture** — 12-layer transformer decoder with causal masking
2. **Training Pipeline** — End-to-end pipeline from raw text → tokens → training
3. **Inference** — Text generation using top-p sampling

It demonstrates how modern language models work and can be trained and used on modest hardware (RTX 2060, 6GB VRAM).

---

## What Each File Does

### Core Model Files

**`GPT/model.py`** — Main language model class
- Defines the `Onyx` class: complete transformer architecture
- Handles token + positional embeddings
- Stacks 12 transformer blocks
- Outputs logits for next-token prediction
- Includes `generate()` method for autoregressive text generation with top-p sampling

**`GPT/TransformerBlock.py`** — Single transformer decoder block
- Multi-head self-attention (12 heads)
- Feed-forward network (2-layer MLP with GELU activation)
- Layer normalization (pre-norm)
- Residual connections with dropout

### Data & Training Files

**`GPT/Dataloader.py`** — Dataset class for training
- `GPTDataset` class: loads pre-tokenized text from `.npy` files
- Creates overlapping (input, target) pairs using a sliding window
- Input: tokens at positions [i, i+seq_length)
- Target: tokens at positions [i+1, i+seq_length+1) — shifted by 1 for next-token prediction
- Memory-mapped loading for efficient RAM usage

**`GPT/precompute.py`** — Tokenization pipeline
- Downloads raw text data
- Tokenizes using GPT-2 tokenizer (50k vocab)
- Saves tokenized arrays as `.npy` files for fast loading
- Run this **once** before training to prepare data

**`GPT/train.py`** — Training loop
- Loads dataset and creates DataLoader
- Implements full training loop with mixed precision (AMP)
- Gradient accumulation (simulates larger batches on limited VRAM)
- Gradient clipping to prevent exploding gradients
- Validation after each epoch
- Saves best model checkpoint when validation accuracy improves

---

## Requirements

### Software

- **Python 3.10+** (uses modern type hints like `str | Path`)
- **PyTorch** (with CUDA support recommended)
- **Transformers** (HuggingFace library)
- **NumPy, tqdm, datasets**

### Installation

```bash
# Install PyTorch with CUDA support (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install transformers datasets tqdm numpy
```

### Hardware

**Tested on:**
- RTX 2060 (6GB) — Works with batch_size=2, seq_length=1024, mixed precision
---

## How It Works

### The Big Picture

```
Raw Text Data
    ↓ (precompute.py)
Token IDs (.npy files)
    ↓ (Dataloader.py)
Batches of (input, target) pairs
    ↓ (train.py)
Model (model.py) computes logits
    ↓
Cross-entropy loss
    ↓
Backprop → Optimizer step
    ↓
Better model weights
    ↓ (generate())
Generated text from prompt
```

### Step 1: Tokenization (`precompute.py`)

Raw text needs to be converted to numbers before the model can process it.

```bash
python GPT/precompute.py
```

**What happens:**
- Loads raw text from a data source
- Uses GPT-2 tokenizer to convert text → token IDs (integers 0-50,256)
- Saves as `.npy` files (numpy arrays) for fast disk loading

**Example:**
```
"Hello world" → [15496, 995]  # Token IDs
```

**Output files:**
- `Data/wikitext_train.npy` — Training tokens
- `Data/wikitext_val.npy` — Validation tokens

---

### Step 2: Creating Training Pairs (`Dataloader.py`)

The model learns to predict the next token. To train it, we need (input, target) pairs:

```python
from GPT.Dataloader import GPTDataset

dataset = GPTDataset("Data/wikitext_train.npy", seq_length=1024, stride=256)

x, y = dataset[0]
# x = tokens[0:1024]        → what the model sees
# y = tokens[1:1025]        → what it should predict (shifted by 1)
```

**Sliding window mechanism:**
- Takes 1024-token windows from the full token sequence
- Steps forward by 256 tokens each sample (75% overlap)
- Creates thousands of training examples from one long sequence

**Why offset by 1?**
- The model learns: "Given these tokens, predict the NEXT one"
- This is called "next-token prediction"
- It's how all language models are trained

---

### Step 3: Training (`train.py`)

The training loop repeatedly:

1. **Load a batch** of (input, target) pairs
2. **Forward pass** — model predicts logits for each position
3. **Compute loss** — how wrong are the predictions? (cross-entropy)
4. **Backward pass** — compute gradients
5. **Optimizer step** — adjust weights to reduce loss
6. **Repeat**

```bash
python GPT/train.py
```

**Key optimizations for limited VRAM:**

- **Gradient Accumulation**: Simulate batch_size=16 using batch_size=2 (8 steps)
- **Mixed Precision (AMP)**: Use FP16 for computation (faster, less memory), FP32 for gradients (stable)
- **Gradient Clipping**: Prevent weight explosions

**Output:**
```
[Epoch 1/2] Train Loss: 4.8234 | Train Acc: 22.45% | Val Loss: 4.2156 | Val Acc: 31.89%  👍  SAVED
[Epoch 2/2] Train Loss: 3.9876 | Train Acc: 38.67% | Val Loss: 3.7234 | Val Acc: 43.39%  👍  SAVED
```

---

### Step 4: Generation (`model.generate()`)

Once trained, generate text by repeatedly predicting the next token:

```python
from GPT.model import Onyx
import torch

model = Onyx()
model.load_state_dict(torch.load("GPT/best_model.pth"))
model.eval()

with torch.no_grad():
    text = model.generate("Once upon a time", max_tokens=100, top_p=0.9, temp=0.7)
    print(text)
```

**How it works:**
1. Tokenize the prompt → token IDs
2. Run model to get logits for next position
3. Sample next token using **top-p sampling** (nucleus sampling)
4. Append token and repeat
5. Stop after `max_tokens` or end-of-sequence token

**Top-P Sampling:**
- Not greedy (always picking the highest probability token)
- Not fully random (picking from the whole distribution)
- Instead: pick from the top tokens that sum to `top_p` probability
- Makes output coherent yet diverse

---

## Architecture

### Model Components

```
Input tokens (token IDs)
    ↓
Embedding Layer
    ├─ Token embedding: maps ID → 768-dim vector
    └─ Positional embedding: encodes position in sequence
    ↓
Stack of 12 Transformer Blocks
    ├─ Multi-Head Self-Attention
    │   └─ Causal mask (prevents attending to future tokens)
    ├─ Residual connection + dropout
    ├─ Feed-Forward Network (2-layer MLP)
    └─ Residual connection + dropout
    ↓
Final LayerNorm
    ↓
Linear projection to vocab size (50,257)
    ↓
Output logits
```

### Transformer Block in Detail

Each of 12 blocks does:

```
Input X
    ↓
LayerNorm → Multi-Head Attention → Dropout → Add X (residual)
    ↓
LayerNorm → FFN (Dense→GELU→Dense) → Dropout → Add (residual)
    ↓
Output
```

**Multi-Head Attention:**
- 12 independent attention heads
- Each head attends to different aspects of the input
- Results concatenated and projected

**Causal Mask:**
- Prevents attending to future positions
- Implemented as upper-triangular matrix of `-inf` values
- Ensures the model can't "cheat" by looking ahead during training

---

## Configuration

### Model Hyperparameters (in `train.py`)

| Parameter | Value | What It Controls |
|-----------|-------|------------------|
| `VOCAB_SIZE` | 50,257 | Number of unique tokens (GPT-2 vocab) |
| `CONTEXT_LEN` | 1024 | Max sequence length the model can process |
| `EMB_SIZE` | 768 | Embedding dimension (d_model in papers) |
| `NUM_HEADS` | 12 | Number of attention heads |
| `NUM_LAYERS` | 12 | Number of transformer blocks stacked |
| `DROPOUT` | 0.1 | Regularization (prevents overfitting) |

### Training Hyperparameters (in `train.py`)

| Parameter | Value | What It Controls |
|-----------|-------|------------------|
| `BATCH_SIZE` | 2 | Samples per GPU per step |
| `ACCUMULATION_STEPS` | 8 | Effective batch = 2 × 8 = 16 |
| `EPOCHS` | 2 | Times to iterate through dataset |
| `LR` | 1e-4 | Learning rate (AdamW optimizer) |
| `STRIDE` | 256 | Step size in sliding window (controls overlap) |

### Dataset Hyperparameters (in `train.py`)

| Parameter | Value | What It Controls |
|-----------|-------|------------------|
| `CONTEXT_LEN` | 1024 | Sequence length for training |
| `STRIDE` | 256 | Overlap between consecutive sequences |

---

## Adjusting Configuration

### For Slower/Older GPUs (out of memory?)

Reduce any of these:

```python
BATCH_SIZE = 1              # Was 2
ACCUMULATION_STEPS = 16     # Increase to keep effective batch ≈ same

# OR reduce model size:
EMB_SIZE = 512              # Was 768
NUM_LAYERS = 6              # Was 12
NUM_HEADS = 6               # Was 12 (must divide EMB_SIZE)
```

### For Faster GPUs (want better results?)

Increase training:

```python
EPOCHS = 5                  # Was 2 (longer training = better)
BATCH_SIZE = 8              # Was 2 (if VRAM allows)
ACCUMULATION_STEPS = 2      # Was 8

# Or train longer on same model:
EPOCHS = 20
```

### For Longer Context Windows

```python
CONTEXT_LEN = 2048          # Was 1024
STRIDE = 512                # Adjust accordingly
```

---

## Troubleshooting

### `RuntimeError: CUDA out of memory`

Reduce memory usage:

1. **Reduce batch size:**
   ```python
   BATCH_SIZE = 1
   ACCUMULATION_STEPS = 16  # Keep effective batch = 16
   ```

2. **Reduce sequence length:**
   ```python
   CONTEXT_LEN = 512        # Was 1024
   ```

3. **Reduce model size:**
   ```python
   EMB_SIZE = 512           # Was 768
   NUM_LAYERS = 8           # Was 12
   ```

4. **Use CPU (slow but works):**
   ```python
   DEVICE = 'cpu'  # In train.py
   ```

### `FileNotFoundError: wikitext_train.npy`

Tokenize the data first:

```bash
python GPT/precompute.py
```

This creates the `.npy` files that training needs.

### Training loss doesn't decrease

**Checklist:**
- Learning rate too high? Try `LR = 5e-5`
- Data loading correct? Verify `x.shape == (batch_size, seq_len)` and `y.shape == (batch_size, seq_len)`
- Causal mask correct? Should prevent attention to future positions
- Delete corrupted checkpoint: `rm GPT/best_model.pth` and restart

### Model generates nonsense

This is normal for untrained models. After 2 epochs of training, generation quality improves significantly.

For better results, train longer:

```python
EPOCHS = 5  # or higher
```

---

## Quick Reference

### Training

```bash
python GPT/precompute.py    # Tokenize data (one-time)
python GPT/train.py         # Train the model
```

### Inference

```python
from GPT.model import Onyx
import torch

model = Onyx()
model.load_state_dict(torch.load("GPT/best_model.pth"))
model.eval()

with torch.no_grad():
    text = model.generate("Your prompt here", max_tokens=100)
    print(text)
```

### Check CUDA

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Radford et al.
- [PyTorch Documentation](https://pytorch.org/docs/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)

---

## License

MIT License