"""Reuse a local dataset or download the public Kaggle dataset once."""
from pathlib import Path
import shutil

from schema import ROOT


def download(root: Path = ROOT) -> Path:
    destination = root / "data/raw/water_potability.csv"
    if destination.is_file():
        print(f"Using {destination}")
        return destination
    import kagglehub

    source_dir = Path(kagglehub.dataset_download("adityakadiwal/water-potability"))
    source = next(source_dir.rglob("water_potability.csv"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    print(f"Downloaded {destination}")
    return destination


if __name__ == "__main__":
    download()
