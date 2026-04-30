import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import GPT2Tokenizer
from datasets import load_dataset

PARENT = Path("G:\\Projects\\Python\\Onyx")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

dataset = load_dataset("wikitext", "wikitext-103-v1")
train_text = dataset['train']['text']
val_text = dataset["validation"]["text"]

def tokenize_batch(text_list: list) -> np.ndarray:
    tokens = []
    for txt in tqdm(text_list, desc="Tokenizing", unit="line"):
        if txt.strip():
            encoded = tokenizer.encode(txt)
            tokens.extend(encoded)
    return np.array(tokens, dtype=np.int32)

print("Processing training set...")
train_tokens = tokenize_batch(train_text)
np.save(PARENT / "Data\\wikitext_train.npy", train_tokens)

print("Processing validation set...")
val_tokens = tokenize_batch(val_text)
np.save(PARENT / "Data\\wikitext_val.npy", val_tokens)

print("-" * 30)
print(f"Tokenization Complete!")
print(f"Train size: {len(train_tokens):,} tokens")
print(f"Val size:   {len(val_tokens):,} tokens")