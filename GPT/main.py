from pathlib import Path
from model import Onyx
import torch

PARENT = Path(r"G:\\Projects\\Python\\Onyx")
MODEL_SAVE_PATH = PARENT / Path(r'best_model.pth')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create fresh Onyx model
model = Onyx().to(device)

model.load_state_dict(torch.load(MODEL_SAVE_PATH), strict=True)

# Test inference
model.eval()
with torch.no_grad():
    prompt = input("User: ").strip()
    # Test generation
    print(f"\nModel:")
    output = model.generate(prompt, max_tokens=50)
    print(output)
