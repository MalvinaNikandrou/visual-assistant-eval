# Multilingual Image Question Answering

The multilingual image question answering data can be found under `tasks/multilingual_image_question_answering/data`. This contains the QA data translated in 34 languages in the following structure:

```
tasks/multilingual_image_question_answering/data/
    ├── val_subsample_ar.json          # Arabic
    ├── val_subsample_bn.json          # Bengali
    ├── val_subsample_cs.json          # Czech
    └── ...                      # 32 more languages
```


## Download the images from the original VizWiz dataset 

Source: (https://vizwiz.org/tasks-and-datasets/vqa/)[https://vizwiz.org/tasks-and-datasets/vqa/]

```bash
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip

unzip val.zip -d tasks/culture_image_captioning/data/images/

rm val.zip
```

## To re-generate the translated data

1. Download the [VizWiz VQA data](https://vizwiz.org/tasks-and-datasets/vqa/):

```bash
wget https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip
unzip Annotations.zip -d tasks/multilingual_image_question_answering/data/
rm Annotations.zip
# Run translation pipeline
bash tasks/multilingual_image_question_answering/src/script.sh
```

## Results analysis

The aggregation script:

```bash
python tasks/multilingual_image_question_answering/src/average_results.py
```

provides accuracy results:
- **Per-language accuracy**: Individual performance for each of 34 languages
- **Per-script accuracy**: Performance grouped by writing system (Latin, Arabic, etc.)
- **Per-resource accuracy**: Performance by high/medium/low resource categories