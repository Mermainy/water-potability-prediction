import os
import kagglehub
from pathlib import Path


PATH = Path.cwd()
DATA_PATH = PATH / "data" / "raw"
os.makedirs(DATA_PATH, exist_ok=True)
dataset = "adityakadiwal/water-potability"
kagglehub.dataset_download(dataset, output_dir=str(DATA_PATH))