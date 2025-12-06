# Image Captioning Data

## Download VizWiz images and annotations

Source: (https://vizwiz.org/tasks-and-datasets/image-captioning/)[https://vizwiz.org/tasks-and-datasets/image-captioning/]

Run the following commands to download and extract the images and annotations:

```bash
# Download images
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip

# Download annotations
wget https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip

# Extract files
unzip train.zip -d tasks/culture_image_captioning/data/images/
unzip val.zip -d tasks/culture_image_captioning/data/images/
unzip annotations.zip -d tasks/culture_image_captioning/data/

# Clean up
rm train.zip val.zip annotations.zip
```

## Prepare the cultural images

Extract the subset of images with cultural annotations:

```bash
python tasks/culture_image_captioning/prepare_vizwiz_culture_images.py
```

This creates a separate folder with the 324 culturally-relevant images at:
`tasks/culture_image_captioning/data/images/culture_images/`

## Data structure

After setup, the data directory will contain:
```
data/
├── images/
│   ├── train/          # Training images (not used in evaluation)
│   ├── val/            # Validation images (500 images)
│   └── culture_images/ # Cultural subset (324 images)
└── annotations/
    ├── train.json          # Training annotations
    ├── val.json            # Validation annotations (original captions)
    └── vizwiz_culture.csv  # Culture-aware annotations  
```


## Evaluation

To evaluate generated captions, we use the origial RefClipScore [implementation](https://github.com/jmhessel/clipscore). The script expects model outputs in JSON format with image IDs and generated captions.


An example of preparing the references and generations for RefClipScore evaluation is provided in `prepare_for_refclipscore.py`.
```bash
python tasks/culture_image_captioning/prepare_for_refclipscore.py \
    --generations_path path/to/your/generations.json \
    --generations_output_path path/to/refclipscore/data
```

## References

- Original VizWiz dataset: [https://vizwiz.org/tasks-and-datasets/image-captioning/](https://vizwiz.org/tasks-and-datasets/image-captioning/)
- RefCLIPScore implementation: [https://github.com/jmhessel/clipscore](https://github.com/jmhessel/clipscore)
- Cultural annotations: [Karamolegkou et al., 2024](https://aclanthology.org/2024.hucllm-1.5/)