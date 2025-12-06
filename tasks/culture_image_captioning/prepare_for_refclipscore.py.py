import argparse
import json
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def load_generations_from_csv(generations_csv):
    """Expected heading: [image_path, prompt, ground_truth, response]

    Columns are separated by tabs.
    Keep only the image basename in the image_path column.
    """
    generations = pd.read_csv(generations_csv, sep="\t", quoting=3, escapechar="\\", quotechar='"')
    generations["image_path"] = generations["image_path"].apply(os.path.basename)
    return generations


def convert_generations_to_json(generations_csv, generations_json):
    """Convert a CSV file to a JSON file.

    The format of the json file should be:
    {
        "image_path": {
            "response": "response"
        }
    }
    """
    generations = load_generations_from_csv(generations_csv)
    generations_dict = generations.set_index("image_path")["response"].to_dict()
    with open(generations_json, "w") as f:
        json.dump(generations_dict, f)


def convert_groundtruths_to_json(annotations_json, groundtruths_json):
    """Convert the references to the correct format.

    The format of the final json file should be:
    {
        "image_path": {
            "references": ["reference1", "reference2", ...]
        }
    }
    """
    with open(annotations_json) as f:
        annotations = json.load(f)
    image_id_to_images = {image["id"]: image["file_name"] for image in annotations["images"]}
    image_id_to_annotations = {image_id_to_images[image["image_id"]]: [] for image in annotations["annotations"]}
    for image in annotations["annotations"]:
        image_id_to_annotations[image_id_to_images[image["image_id"]]].append(image["caption"])

    with open(groundtruths_json, "w") as f:
        json.dump(image_id_to_annotations, f)


def convert_groundtruths_csv_to_json(annotations_csv, groundtruths_json):
    """Convert the references to the correct format.

    The format of the final json file should be:
    {
        "image_path": {
            "references": ["reference1", "reference2", ...]
        }
    }
    """
    annotations = pd.read_csv(annotations_csv)
    # get image, corrected caption and corrected caption 2
    image_id_to_annotations = {}
    for index, row in annotations.iterrows():
        image_id = row["image"]
        caption = row["corrected_caption"]
        caption2 = row["corrected_caption2"]
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(caption)
        image_id_to_annotations[image_id].append(caption2)

    with open(groundtruths_json, "w") as f:
        json.dump(image_id_to_annotations, f)


def main(args):
    # Prepare the references
    logger.info("Preparing reference captions...")
    convert_groundtruths_to_json(
        "tasks/data/culture_image_captioning/annotations/val.json",
        "vizwiz_captioning_references.json",
    )
    convert_groundtruths_csv_to_json(
        "tasks/data/culture_image_captioning/annotations/vizwiz_culture.csv",
        "vizwiz_culture_captioning_references.json",
    )
    # Prepare the generations
    logger.info("Preparing generated captions...")
    convert_generations_to_json(args.generations_path, args.generations_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generations_path",
        type=str,
        required=False,
        help="Path to the model generationed captions CSV file.",
    )
    parser.add_argument(
        "--generations_output_path",
        type=str,
        required=False,
        help="Path to the generations JSON output file.",
    )
    args = parser.parse_args()
    main(args)
