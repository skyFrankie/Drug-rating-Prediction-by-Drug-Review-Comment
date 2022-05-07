import os
from pathlib import Path


DATA_PATH = Path(r"C:\Users\ftsky\PycharmProjects\COMP5434_project\submission_V2\data_source")
OUTPUT_PATH = Path(
    rf"C:\Users\ftsky\PycharmProjects\COMP5434_project\submission_V2\output"
)
TRANSFORMER_PATH = OUTPUT_PATH / "Transformer"

if not os.path.exists(TRANSFORMER_PATH):
    os.makedirs(TRANSFORMER_PATH)
    print("Directory created for transformer.")
