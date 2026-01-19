import torch
from pathlib import Path

# Make sure this path is correct from the root of your project
model_path = Path("notebooks/lstm_model_inference.pth")

if not model_path.exists():
    print(f"ERROR: File does not exist at {model_path.resolve()}")
else:
    print(f"File found at {model_path.resolve()}")
    try:
        # Load checkpoint onto CPU to avoid GPU issues
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        print("\n--- Checkpoint Contents ---")
        print("Keys found:", checkpoint.keys())
        
        print("\n--- Model Hyperparameters ---")
        print(f"Input Dim: {checkpoint.get('input_dim')}")
        print(f"Hidden Dim: {checkpoint.get('hidden_dim')}")
        print(f"Num Layers: {checkpoint.get('num_layers')}")

        print("\n--- Normalization Stats ---")
        stats = checkpoint.get('normalization_stats', {})
        mean = stats.get('mean')
        std = stats.get('std')
        print(f"Mean tensor shape: {mean.shape if hasattr(mean, 'shape') else 'Not found'}")
        print(f"Std tensor shape: {std.shape if hasattr(std, 'shape') else 'Not found'}")

        print("\nSUCCESS: Checkpoint loaded and inspected.")

    except Exception as e:
        print(f"\nFAILED to load or inspect checkpoint. Error: {e}")