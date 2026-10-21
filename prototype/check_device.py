"""Report local PyTorch/MPS readiness for ModelGuard's Qwen baseline."""
import platform
import sys

print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print(f"Machine architecture: {platform.machine()}")

try:
    import torch
except ImportError:
    raise SystemExit("PyTorch not installed; run pip install -r prototype/requirements.txt")

print(f"PyTorch: {torch.__version__}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")

if platform.system() == "Darwin" and platform.machine() != "arm64":
    print("WARNING: You are running non-native Python (Rosetta/x86_64). Install native arm64 Python.")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    try:
        matrix = torch.eye(4, dtype=torch.float32, device=device)
        result = (matrix @ matrix).sum().item()
        print(f"MPS float32 matmul: PASS (sum={result})")
        print("Suggested training command: python prototype/train.py --device mps --max-steps 2")
    except Exception as exc:
        print(f"MPS float32 matmul: FAIL: {exc}")
        raise SystemExit(2) from exc
else:
    print("MPS not ready. On Apple Silicon, verify macOS and the installed PyTorch wheel.")
    print("You can still test CPU: python prototype/train.py --device cpu --max-steps 1")
