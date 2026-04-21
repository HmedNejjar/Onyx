#type: ignore
PARENT = "G:\\Projects\\Python\\Onyx"
import sys
sys.path.insert(2, PARENT)

import numpy as np
from pathlib import Path
from tqdm import tqdm  # Progress bar for long-running tokenization
from Tokenizer.Encoder import BPE
from datasets import load_dataset

# Configuration for paths
TOKENIZER_PATH = PARENT / Path(r'Tokenizer\BPE_30k.json')

# Initialize and load the pre-trained BPE tokenizer
tokenizer = BPE()
tokenizer.load(TOKENIZER_PATH)

# Load WikiText-103; 'train' is large (~100M tokens), so progress bars are helpful
dataset = load_dataset("wikitext", "wikitext-103-v1")
train_text = dataset['train']['text']    
val_text = dataset["validation"]["text"]

def tokenize_batch(text_list: list, tokenizer: BPE) -> np.ndarray:
    """
    Encodes a list of strings into a flat numpy array of token IDs.
    """
    tokens = []
    
    # Wrap text_list with tqdm for a visual progress bar in the console
    for txt in tqdm(text_list, desc="Tokenizing", unit="line"):
        if txt.strip():  # Skip empty lines or whitespace-only strings
            encoded = tokenizer.encode(txt)
            tokens.extend(encoded)
            
    return np.array(tokens, dtype=np.int32)

# --- Process Training Data ---
print("Processing training set...")
train_tokens = tokenize_batch(train_text, tokenizer)
# Save as a binary file for fast loading during model training
np.save("G:\\Projects\\Python\Onyx\\Data\\wikitext_train.npy", train_tokens)

# --- Process Validation Data ---
print("Processing validation set...")
val_tokens = tokenize_batch(val_text, tokenizer)
np.save("G:\\Projects\\Python\Onyx\\Data\\wikitext_val.npy", val_tokens)

# Summary of the resulting dataset size
print("-" * 30)
print(f"Tokenization Complete!")
print(f"Train size: {len(train_tokens):,} tokens")
print(f"Val size:   {len(val_tokens):,} tokens")