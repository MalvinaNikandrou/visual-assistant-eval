# Evaluating Multimodal Language Models as Visual Assistants for Visually Impaired Users

<!-- [![ACL 2025](https://img.shields.io/badge/ACL-2025-blue)](https://2025.aclweb.org/) -->
[![Paper](https://img.shields.io/badge/Paper-ACL_2025-red)](https://arxiv.org/abs/2503.22610)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## 📊 Tasks and Datasets

We present a comprehensive evaluation framework informed by a user survey to identify
adoption patterns and key challenges visually impaired users face with LLM technologies. Our framework consists of five key tasks:

### 1. Culture-Aware Image Captioning
Evaluating cultural understanding in image descriptions 
- **Dataset**: [VizWiz](https://vizwiz.org/) images with two evaluation settings:
  - **Original**: 500 images with standard captions ([Gurari et al.,
2020](https://link.springer.com/chapter/10.1007/978-3-030-58520-4_25))
  - **Cultural**: 324 images with culture-aware captions ([Karamolegkou et al., 2024](https://aclanthology.org/2024.hucllm-1.5/))
- **Metric**: RefCLIPScore
- **Setup**: [See task README](tasks/culture_image_captioning/README.md)

### 2. Multilingual Image Question Answering  
VQA across 35 languages
- **Dataset**: 500 English VizWiz VQA samples ([Gurari et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Gurari_VizWiz_Grand_Challenge_CVPR_2018_paper.html)) translated to 34 languages
- **Languages**: High (21), Medium (10), and Low-resource (4) languages
- **Metric**: VQA Accuracy
- **Setup**: [See task README](tasks/multilingual_image_question_answering/README.md)

### 3. Optical Braille Recognition
Transcribing and answering questions about Braille text
- **Dataset**: Novel Braille transcription and QA datasets
- **Tasks**: Sentence transcription, Cross-script QA
- **Metrics**: chrF++, F1-score
- **Setup**: [See task README](tasks/obr/README.md)

### 4. Video Object Recognition
Identifying general and assistive objects in videos
- **Dataset**: 1,036 ORBIT videos ([Massiceti et al., 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Massiceti_ORBIT_A_Real-World_Few-Shot_Dataset_for_Teachable_Object_Recognition_ICCV_2021_paper.pdf)) featuring 880 general, 156 assistive objects
- **Metric**: LAVE accuracy
- **Setup**: [See task README](tasks/video_recognition_and_question_answering/README.md)

### 5. Video Question Answering
Answering descriptive, spatial, and adversarial questions
- **Dataset**: 882 questions covering descriptive, spatial, and adversarial questions
- **Metric**: LAVE accuracy
- **Setup**: [See task README](tasks/video_recognition_and_question_answering/README.md)


### Data Access

All evaluation data is organized under the `tasks/` directory:
```
tasks/
├── culture_image_captioning/          # Culture-aware image descriptions
├── multilingual_image_question_answering/  # VQA in 35 languages
├── obr/                               # Optical Braille Recognition
└── video_recognition_and_question_answering/  # Video understanding tasks
```

## 🤖 Model Inference

The inference code is based on [https://github.com/coastalcph/vizwiz-culture](https://github.com/coastalcph/vizwiz-culture), which runs runs image-text inference with SOTA vision-language models.


### Installation

1. Basic install
```bash
conda create -n visassistant python=3.10
conda activate visassistant
pip install -e .
```

2. (Optional) Install flash-attention

```bash
pip install flash-attn --no-build-isolation

# Verify import; if output is empty installation was successful
python -c "import torch; import flash_attn_2_cuda"
```


### Example commands:

#### Image Tasks
```bash
# LLaVA-1.6 on VizWiz VQA
python run.py \
  model=llava \
  dataset=vizwiz_vqa \
  dataset.path=data/val.json \
  dataset.images_path=data/val

# Qwen2-VL on culture captioning  
python run.py \
  model=qwen2-vl \
  dataset=captioning \
  dataset.path=data/culture_images
```

#### Video Tasks
```bash
# InternVL2.5 on video object recognition
python run.py \
  model=internvl2.5-video \
  dataset=orbit_vqa \
  dataset.path=tasks/video_object_recognition/orbit_recognition_question_answers.json \
  dataset.images_path=tasks/video_object_recognition/orbit_videos
```

#### Multilingual Evaluation
```bash
# Run on all languages
./scripts/multilingual_image_question_answering.sh llava
```

See model-specific examples in the [inference section](#supported-models) below.



## 📚 Citation

If you use this evaluation framework or datasets, please cite:

```bibtex
@inproceedings{karamolegkou-etal-2025-evaluating,
    title = "Evaluating Multimodal Language Models as Visual Assistants for Visually Impaired Users",
    author = "Karamolegkou, Antonia  and
      Nikandrou, Malvina  and
      Pantazopoulos, Georgios  and
      Sanchez Villegas, Danae  and
      Rust, Phillip  and
      Dhar, Ruchira  and
      Hershcovich, Daniel  and
      S{\o}gaard, Anders",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1260/",
    doi = "10.18653/v1/2025.acl-long.1260",
    pages = "25949--25982",
    ISBN = "979-8-89176-251-0"
}
```
