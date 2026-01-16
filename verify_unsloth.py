import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"Torch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"Torch Check Failed: {e}")

try:
    from unsloth import FastLanguageModel
    print("Unsloth: Successfully imported FastLanguageModel")
except ImportError as e:
    print(f"Unsloth Import Failed: {e}")
except Exception as e:
    print(f"Unsloth Init Failed: {e}")
