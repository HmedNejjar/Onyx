from model import Onyx
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from Dataloader import GPTDataset
from pathlib import Path
from tqdm import tqdm

PARENT = Path(r"G:\\Projects\\Python\\Onyx")
TRAIN_FILE_PATH = PARENT / Path(r'Data/wikitext_train.npy')
VAL_FILE_PATH = PARENT / Path(r'Data/wikitext_val.npy')
MODEL_SAVE_PATH = PARENT / Path(r'best_model.pth')

VOCAB_SIZE = 30_000
STRIDE = 256
CONTEXT_LEN = 512
EMB_SIZE = 512
NUM_HEADS = 16
NUM_LAYERS = 12
DROPOUT = 0.1
BATCH_SIZE = 2
EPOCHS = 10
LR = 1e-4

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {DEVICE.upper()} for training')

# Move model to device BEFORE creating optimizer so optimizer holds GPU parameters
model = Onyx(VOCAB_SIZE, CONTEXT_LEN, EMB_SIZE, NUM_HEADS, NUM_LAYERS, DROPOUT).to(DEVICE)

if MODEL_SAVE_PATH.exists():
    print(f"Loading model from {MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

optimizer = AdamW(model.parameters(), LR, weight_decay=1e-5)
loss_fn = CrossEntropyLoss()

train_dataset = GPTDataset(TRAIN_FILE_PATH, CONTEXT_LEN, STRIDE)
val_dataset   = GPTDataset(VAL_FILE_PATH,   CONTEXT_LEN, STRIDE)

# num_workers speeds up data loading; pin_memory speeds up CPU→GPU transfers
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute token-level accuracy: percentage of correctly predicted next tokens.

    Args:
        logits (torch.Tensor): Model output of shape [batch_size, seq_length, vocab_size]
        targets (torch.Tensor): Target tokens of shape [batch_size, seq_length]

    Returns:
        float: Accuracy as a percentage (0-100)
    """
    predictions = torch.argmax(logits, dim=-1)
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
        for x, y in tqdm(val_loader, desc="  Validating", leave=False):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            logits = model(x)
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))

            total_loss     += loss.item()
            total_accuracy += accuracy(logits, y)
            num_batches    += 1

    return total_loss / num_batches, total_accuracy / num_batches

def train_one_epoch(epoch: int, best_accuracy: float) -> float:
    """
    Train for a single epoch and validate.
    
    Args:
        epoch (int): Current epoch number (0-indexed)
        best_accuracy (float): Best validation accuracy seen so far
    
    Returns:
        float: Updated best_accuracy if validation improved, else unchanged
    """
    model.train()
    epoch_loss     = 0.0
    epoch_accuracy = 0.0
    num_batches    = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for x, y in pbar:
        # Move batch data to the selected device.
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        
        # Forward pass: compute logits from model input.
        logits = model(x)
        # Compute cross-entropy loss over the flattened batch/sequence output.
        loss   = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))

        # Zero gradients, backpropagate, and update parameters.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute batch accuracy and accumulate metrics.
        acc             = accuracy(logits, y)
        epoch_loss     += loss.item()
        epoch_accuracy += acc
        num_batches    += 1

        # Update progress bar with the current batch metrics.
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}%")

    avg_train_loss = epoch_loss     / num_batches
    avg_train_acc  = epoch_accuracy / num_batches

    # Validate on the held-out dataset after each epoch.
    val_loss, val_acc = validate()

    # Save the model when validation accuracy improves.
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%  👍  SAVED")
    else:
        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    return best_accuracy


if __name__ == "__main__":
    best_accuracy = 0.0
    for epoch in range(EPOCHS):
        best_accuracy = train_one_epoch(epoch, best_accuracy)