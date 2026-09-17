import sys
from pathlib import Path

# Keep the existing code/ layout without importing Python's stdlib code module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
