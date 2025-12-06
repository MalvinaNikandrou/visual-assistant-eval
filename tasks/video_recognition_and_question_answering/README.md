# Video Object Recognition & Question Answering

This directory contains the data and metric code for evaluating multimodal language models on video understanding tasks using the [ORBIT dataset](https://github.com/microsoft/ORBIT-Dataset/tree/master).

## Data format

The video question-answer pairs for the categories object recognition (O), descriptive (D), spatial (S), adversarial (S) are in `tasks/video_recognition_and_question_answering/data/combined_video_qa_data.json`, with the following fields:

| Field            | Type      | Description |
|------------------|-----------|-------------|
| `id`             | string    | Unique identifier for the video. |
| `video_path`     | string    | Path to the associated video file. |
| `question`       | string    | Question about the video/object. |
| `question_type`  | string    | Encoded question category from {O, D, S, A}. |
| `answer`         | string    | Ground-truth answer. |
| `object`         | string    | Object annotation from the original dataset (e.g, "white mobility cane"). |
| `group`          | string    | Group of the object class (e.g., cane). |
| `is_assistive_object`  | boolean   | Whether the object belongs to the subset of assistive objects. |
| `video_type`     | string    | Type of video scenario (clean or clutter). |


## Evaluation

Evaluate the predictions using the LAVE accuracy, for example:

```
RESULTS_FILE="tasks/video_recognition_and_question_answering/results/Qwen/Qwen2-VL-7b-Instruct-outputs.csv"

# Run LAVE evaluation
python tasks/video_recognition_and_question_answering/src/lave_accuracy.py \
    --data_file $RESULTS_FILE \
    --max_new_tokens 256 \
    --load_in_8bit
```

The evaluation script will:
1. Use Llama-3.3-70B to judge each prediction
2. Generate ratings on a 1-3 scale (normalized to 0-1)
3. Print the overall, non-assistive object, and assistive object accuracy
4. Save results to `{RESULTS_FILE}-outputs-acc.csv`