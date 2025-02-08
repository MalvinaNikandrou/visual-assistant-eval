python run.py \
  model=qwen2-vl-video  \
  generation_config=hf_vqa  \
  model.json_mode=false \
  dataset.path=tasks/orbit_video_question_answering/data/combined_qa_data.json \
  +dataset.images_path=tasks/orbit_video_question_answering/data/videos  \
  dataset.template_name=qwen2vl_video_qa \
  output_path=tasks/orbit_video_question_answering/results \
  _callback_dict.wandb.project=orbit_vqa \
  dataset=orbit_vqa

python tasks/orbit_video_question_answering/src/lave_accuracy.py \
   --data_file tasks/orbit_video_question_answering/results/Qwen/Qwen2-VL-7b-Instruct-outputs.csv

python run.py \
  model=llava-next-video  \
  generation_config=hf_vqa  \
  model.json_mode=false \
  dataset.path=tasks/orbit_video_question_answering/data/combined_qa_data.json \
  +dataset.images_path=tasks/orbit_video_question_answering/data/videos  \
  dataset.template_name=llava_next_video_qa \
  output_path=tasks/orbit_video_question_answering/results \
  _callback_dict.wandb.project=orbit_vqa \
  dataset=orbit_vqa

python tasks/orbit_video_question_answering/src/lave_accuracy.py \
   --data_file tasks/orbit_video_question_answering/results/llava-hf/LLaVA-NeXT-Video-7B-hf-outputs.csv

python run.py \
  model=llava-video  \
  generation_config=hf_vqa  \
  model.json_mode=false \
  dataset.path=tasks/orbit_video_question_answering/data/combined_qa_data.json \
  +dataset.images_path=tasks/orbit_video_question_answering/data/videos  \
  dataset.template_name=default_video_qa \
  output_path=tasks/orbit_video_question_answering/results \
  _callback_dict.wandb.project=orbit_vqa \
  dataset=orbit_vqa

CUDA_VISIBLE_DEVICE=1 python tasks/orbit_video_question_answering/src/lave_accuracy.py \
   --data_file tasks/orbit_video_question_answering/results/OpenGVLab/VideoChat-Flash-Qwen2-7B_res448-outputs.csv
CUDA_VISIBLE_DEVICE=1 python tasks/orbit_video_question_answering/src/lave_accuracy.py \
   --data_file tasks/orbit_video_question_answering/results/lmms-lab/LLaVA-Video-7B-Qwen2-outputs.csv

python run.py \
  model=video-chat  \
  generation_config=hf_vqa  \
  model.json_mode=false \
  dataset.path=tasks/orbit_video_question_answering/data/combined_qa_data.json \
  +dataset.images_path=tasks/orbit_video_question_answering/data/videos  \
  dataset.template_name=default_video_qa \
  output_path=tasks/orbit_video_question_answering/results \
  _callback_dict.wandb.project=orbit_vqa \
  dataset=orbit_vqa


python run.py \
  model=minicpm-video  \
  generation_config=hf_vqa  \
  model.json_mode=false \
  dataset.path=tasks/orbit_video_question_answering/data/combined_qa_data.json \
  +dataset.images_path=tasks/orbit_video_question_answering/data/videos  \
  dataset.template_name=default_video_qa \
  output_path=tasks/orbit_video_question_answering/results \
  _callback_dict.wandb.project=orbit_vqa \
  dataset=orbit_vqa

CUDA_VISIBLE_DEVICE=2 python tasks/orbit_video_question_answering/src/lave_accuracy.py \
   --data_file tasks/orbit_video_question_answering/results/openbmb/MiniCPM-V-2_6-outputs.csv

python run.py \
  model=internvl2.5-video  \
  generation_config=hf_vqa  \
  model.json_mode=false \
  dataset.path=tasks/orbit_video_question_answering/data/combined_qa_data.json \
  +dataset.images_path=tasks/orbit_video_question_answering/data/videos  \
  dataset.template_name=default_video_qa \
  output_path=tasks/orbit_video_question_answering/results \
  _callback_dict.wandb.project=orbit_vqa \
  dataset=orbit_vqa

python tasks/orbit_video_question_answering/src/lave_accuracy.py \
   --data_file tasks/orbit_video_question_answering/results/OpenGVLab/InternVL2_5-8B-outputs.csv

python run.py \
  model=phi3-video  \
  generation_config=hf_vqa  \
  model.json_mode=false \
  dataset.path=tasks/orbit_video_question_answering/data/combined_qa_data.json \
  +dataset.images_path=tasks/orbit_video_question_answering/data/videos  \
  dataset.template_name=phi3_video_qa \
  output_path=tasks/orbit_video_question_answering/results \
  _callback_dict.wandb.project=orbit_vqa \
  dataset=orbit_vqa

python tasks/orbit_video_question_answering/src/lave_accuracy.py \
   --data_file tasks/orbit_video_question_answering/results/microsoft/Phi-3.5-vision-instruct-outputs.csv