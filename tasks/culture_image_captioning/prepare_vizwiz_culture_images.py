import shutil
from pathlib import Path

import pandas as pd

data = pd.read_csv(Path("tasks/culture_image_captioning/data/annotations/vizwiz_culture.csv"))
image_root_dir = Path("tasks/culture_image_captioning/data/images")
target_image_root_dir = Path("tasks/culture_image_captioning/data/culture_images")
target_image_root_dir.mkdir(parents=True, exist_ok=True)

for idx, row in data.iterrows():
    if "train" in row["image"]:
        image_path = image_root_dir.joinpath("train", row["image"])
    else:
        image_path = image_root_dir.joinpath("val", row["image"])
    target_image_path = target_image_root_dir.joinpath(row["image"])
    shutil.copy(image_path, target_image_path)
