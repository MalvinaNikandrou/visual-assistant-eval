# Image Captioning Data

## Download the VizWiz images and annotations

Source: (https://vizwiz.org/tasks-and-datasets/image-captioning/)[https://vizwiz.org/tasks-and-datasets/image-captioning/]

Run the following commands to download and extract the images and annotations:

```bash
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip
wget https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip

unzip train.zip -d tasks/culture_image_captioning/data/images/
unzip val.zip -d tasks/culture_image_captioning/data/images/
unzip test.zip -d tasks/culture_image_captioning/data/images/
unzip annotations.zip -d tasks/data/

rm train.zip val.zip test.zip annotations.zip
```

## Prepare the cultural images
Run the `prepare_vizwiz_culture_images.py` script to copy the relevant images to a separate folder.

```bash
python tasks/culture_image_captioning/prepare_vizwiz_culture_images.py
```

The evaluation sets used in the paper are the validation images in `tasks/culture_image_captioning/data/images/val/` that are copied to `tasks/culture_image_captioning/data/images/culture_images/` by the script.


## Evaluation

To evaluate generated captions we use the origial RefClipScore (implementation)[https://github.com/jmhessel/clipscore].

An example of preparing the references and generations for RefClipScore evaluation is provided in `prepare_for_refclipscore.py`.