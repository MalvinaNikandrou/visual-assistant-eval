# For the orbit video dataset let's get some statistics and plots
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load the dataset
qa_dataset_path = "tasks/video_recognition_and_question_answering/data/combined_video_qa_data.json"
videos_dataset_path = Path("tasks/video_recognition_and_question_answering/data/videos")

with open(qa_dataset_path, "r") as f:
    qa_dataset = json.load(f)

# Number of videos
num_videos = len(qa_dataset)
print(f"Number of videos: {num_videos}")

# Number of videos per group
groups = [v["group"] for v in qa_dataset]
print(f"Number of videos per group: {pd.Series(groups).value_counts()}")

# Number of clean videos
clean_videos = [v for v in qa_dataset if v["video_type"] == "clean"]
print(f"Number of clean videos: {len(clean_videos)}")

# Number of participants
for sample in qa_dataset:
    video_path = sample["video_path"]
    participant = video_path.split("-")[0]
    sample["participant"] = participant

participants = [v["participant"] for v in qa_dataset]
print(f"Number of participants: {pd.Series(participants).nunique()}")

# Number of videos per question type
question_types = [v["question_type"] for v in qa_dataset]
print(f"Number of videos per question type: {pd.Series(question_types).value_counts()}")

# Number of assistive videos per question type
assistive_question_types = [v["question_type"] for v in qa_dataset if v["is_assistive_object"]]
print(f"Number of assistive videos per question type: {pd.Series(assistive_question_types).value_counts()}")

# Number of videos per participant
videos_per_participant = pd.Series(participants).value_counts()

# Number of objects
objects = [v["object"] for v in qa_dataset]
print(f"Number of objects: {pd.Series(objects).nunique()}")
objects = [v["group"] for v in qa_dataset]
print(f"Number of groups: {pd.Series(groups).nunique()}")
# Number of assistive videos
assistive_objects = [v["object"] for v in qa_dataset if v["is_assistive_object"]]
assistive_groups = [v["group"] for v in qa_dataset if v["is_assistive_object"]]
mapping = {
    "recorder": "voice recorder",
    "pencil": "Braille stylus",
}
assistive_groups = [mapping.get(v, v).capitalize() for v in assistive_groups]
print(f"Number of assistive videos: {len(assistive_objects)}")
print("assistive Objects", Counter(assistive_objects).most_common(150))
# General objects
non_assistive_objects = [v["object"] for v in qa_dataset if not v["is_assistive_object"]]
non_assistive_groups = [v["group"] for v in qa_dataset if not v["is_assistive_object"]]
print(Counter(non_assistive_objects).most_common(50))

assistive_groups = [mapping.get(v, v).capitalize() for v in assistive_groups]
# sort based on frequency
assistive_group_counts = Counter(assistive_groups).most_common()
# Make a histogram of the groups
plt.figure(figsize=(12, 6))
plt.barh([v[0] for v in assistive_group_counts], [v[1] for v in assistive_group_counts], zorder=3)
plt.yticks(fontsize=16)
plt.title("ORBIT Video Group histogram", fontdict={"fontsize": 18})
# rotate x labels
plt.xlabel("Count", fontdict={"fontsize": 18})
# add count on top of the bars
# for i, v in enumerate(assistive_group_counts):
#     plt.text(i, v[1] + 0.15, str(v[1]), ha="center", va="bottom", fontsize=14)

# add horizontal grid lines
plt.grid(axis="both", zorder=0)
# ylim
plt.tight_layout()
plt.savefig(
    "tasks/video_recognition_and_question_answering/data/orbit_video_group_histogram.pdf",
    dpi=300,
    bbox_inches="tight",
)


# Make a wordcloud

from wordcloud import STOPWORDS, WordCloud

wordcloud = WordCloud(
    width=600,
    height=600,
    background_color="white",
    collocations=False,
    repeat=False,
    max_words=50,
    stopwords=set(STOPWORDS),
    min_font_size=10,
).generate(",".join(assistive_objects))
plt.figure(figsize=(6, 6), facecolor=None)
# plt.subplot(1, 2, 1)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("tasks/video_recognition_and_question_answering/data/orbit_assistive_object_wordcloud.pdf")

# plt.subplot(1, 2, 2)
wordcloud = WordCloud(
    width=600,
    height=600,
    background_color="white",
    max_words=80,
    stopwords=set(STOPWORDS),
    collocations=False,
    repeat=False,
    min_font_size=10,
).generate(",".join(non_assistive_objects))
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("tasks/video_recognition_and_question_answering/data/orbit_object_wordcloud.pdf")


# Duration histogram
def with_opencv(filename):
    import cv2

    video = cv2.VideoCapture(filename)
    duration = video.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / 30
    if duration < 2:
        print(filename)
    return duration, frame_count


frame_counts = []
durations = []
for video in tqdm(videos_dataset_path.iterdir()):
    duration, frame_count = with_opencv(video)
    durations.append(duration)
    frame_counts.append(frame_count)

# remove outliers
frame_counts = [f for f in frame_counts if f < 2000]
durations = [d for d in durations if d < 78]
print(f"Frame counts min: {np.min(frame_counts)}")
print(f"Frame counts mean: {np.mean(frame_counts)}")
print(f"Frame counts median: {np.median(frame_counts)}")
print(f"Frame counts max: {np.max(frame_counts)}")
print(f"Frame counts std: {np.std(frame_counts)}")

print(f"Duration min: {np.min(durations)}")
print(f"Duration mean: {np.mean(durations)}")
print(f"Duration median: {np.median(durations)}")
print(f"Duration max: {np.max(durations)}")
print(f"Duration std: {np.std(durations)}")

plt.figure(figsize=(10, 6))
plt.hist(durations, bins=50, zorder=3)
plt.title("ORBIT Video Duration Histogram", fontdict={"fontsize": 18})
plt.xlabel("Duration (seconds)", fontdict={"fontsize": 18})
plt.ylabel("Count", fontdict={"fontsize": 18})
plt.grid(axis="y", zorder=0)
plt.tight_layout(pad=0)
plt.savefig("tasks/video_recognition_and_question_answering/data/orbit_video_duration_histogram.pdf")
