# Multilingual Image Question Answering

The multilingual image question answering data across 34 languages can be found under `tasks/multilingual_image_question_answering/data`.

## Subsample and translate the VizWiz VQA data

For the data generation code see `tasks/multilingual_image_question_answering/src/script.sh`.

## Aggregating results across languages

To print the results per language, aggregated per script, and per resource level run:

```
python tasks/multilingual_image_question_answering/src/average_results.py
```