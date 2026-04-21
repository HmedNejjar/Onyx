from model import Onyx
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from Dataloader import GPTDataset
from pathlib import Path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {DEVICE.upper()} for training')

PARENT = Path(r"G:\\Projects\\Python\\Onyx")
TRAIN_FILE_PATH = PARENT / Path(r'Data\\wikitext_train.npy')
VAL_FILE_PATH = PARENT / Path(r'Data\\wikitext_val.npy')

VOCAB_SIZE = 30_000
CONTEXT_LEN = 1024
EMB_SIZE = 1024
NUM_HEADS = 16
NUM_LAYERS = 12
DROPOUT = 0.1
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

model = Onyx(VOCAB_SIZE, CONTEXT_LEN, EMB_SIZE, NUM_HEADS, NUM_LAYERS, DROPOUT)
optimizer = AdamW(model.parameters(), LR, weight_decay= 1e-5)
loss_fn = CrossEntropyLoss()

train_dataset, val_dataset = GPTDataset(TRAIN_FILE_PATH), GPTDataset(VAL_FILE_PATH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute token-level accuracy: percentage of correctly predicted next tokens.
    
    Args:
        logits (torch.Tensor): Model output of shape [batch_size, seq_length, vocab_size]
        targets (torch.Tensor): Target tokens of shape [batch_size, seq_length]
    
    Returns:
        float: Accuracy as a percentage (0-100)
    """
    # Get predicted token IDs (argmax over vocab dimension)
    predictions = torch.argmax(logits, dim=-1)
    # Compare predictions to targets and compute mean accuracy
    correct = (predictions == targets).float().mean()
    return (correct * 100).item()


def validate():
    """
    Run validation loop and compute average loss and accuracy on validation set.
    
    Returns:
        tuple[float, float]: (average_loss, average_accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Forward pass
            logits = model(x)
            # Reshape for loss computation: [batch_size * seq_length, vocab_size] and [batch_size * seq_length]
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
            
            # Compute accuracy
            acc = accuracy(logits, y)
            
            total_loss += loss.item()
            total_accuracy += acc
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    return avg_loss, avg_accuracy


def train():
    best_accuracy = 0.0
    model.to(DEVICE)
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # Forward pass
            logits = model(x)
            # Reshape for loss: flatten batch and sequence dimensions
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            acc = accuracy(logits, y)
            epoch_loss += loss.item()
            epoch_accuracy += acc
            num_batches += 1
        
        # Compute epoch averages
        avg_train_loss = epoch_loss / num_batches
        avg_train_acc = epoch_accuracy / num_batches
        
        # Validation
        val_loss, val_acc = validate()
        
        # Save best model based on validation accuracy
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), PARENT / "best_model.pth")
            print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% ⭐ SAVED")
        else:
            print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")


if __name__ == "__main__":
    train()


