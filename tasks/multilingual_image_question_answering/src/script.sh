# Subsample the VizWiz validation dataset
python subsample.py

# Translate the dataset in 35 languages
python translate.py \
    --model_name facebook/nllb-200-distilled-1.3B \
    --dataset_dir tasks/multilingual_image_question_answering/data/val_subsample_en.json \
    --batch_size 16 \
    --gpu 0
