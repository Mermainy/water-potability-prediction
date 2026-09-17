"""Shared feature order and repository paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = "Potability"
FEATURES = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity",
            "Organic_carbon", "Trihalomethanes", "Turbidity"]
