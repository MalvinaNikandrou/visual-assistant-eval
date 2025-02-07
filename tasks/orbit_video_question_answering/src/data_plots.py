# For the orbit video dataset let's get some statistics and plots
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from pathlib import Path
import numpy as np
from tqdm import tqdm


# Load the dataset
qa_dataset_path = (
    "/users/mn2002/sharedscratch/vizwiz-culture/tasks/video_object_recognition/orbit_question_answers.json"
)
videos_dataset_path = Path("/users/mn2002/sharedscratch/vizwiz-culture/tasks/video_object_recognition/orbit_videos")

with open(qa_dataset_path, "r") as f:
    qa_dataset = json.load(f)

# Number of videos
num_videos = len(qa_dataset)
print(f"Number of videos: {num_videos}")

# Number of videos per group
groups = [v["group"] for v in qa_dataset]
print(f"Number of videos per group: {pd.Series(groups).value_counts()}")

# Number of participants
for sample in qa_dataset:
    video_path = sample["video_path"]
    participant = video_path.split("-")[0]
    sample["participant"] = participant

participants = [v["participant"] for v in qa_dataset]
print(f"Number of participants: {pd.Series(participants).nunique()}")

# Number of videos per participant
videos_per_participant = pd.Series(participants).value_counts()

# Number of objects
objects = [v["object"] for v in qa_dataset]
print(f"Number of objects: {pd.Series(objects).nunique()}")
objects = [v["group"] for v in qa_dataset]
print(f"Number of groups: {pd.Series(groups).nunique()}")
# Number of VIP videos
vip_objects = [v["object"] for v in qa_dataset if v["is_vip_object"]]
vip_groups = [v["group"] for v in qa_dataset if v["is_vip_object"]]
mapping = {
    "recorder": "voice recorder",
}
vip_groups = [mapping.get(v, v).capitalize() for v in vip_groups]
print(f"Number of VIP videos: {len(vip_objects)}")
print(Counter(vip_objects).most_common(50))
# General objects
non_vip_objects = [v["object"] for v in qa_dataset if not v["is_vip_object"]]
non_vip_groups = [v["group"] for v in qa_dataset if not v["is_vip_object"]]
print(Counter(non_vip_objects).most_common(50))

# sort based on frequency
vip_group_counts = Counter(vip_groups).most_common()
# Make a histogram of the groups
plt.figure(figsize=(12, 6))
plt.bar([v[0] for v in vip_group_counts], [v[1] for v in vip_group_counts], zorder=3)
plt.xticks(rotation=40, fontsize=16)
plt.title("ORBIT Video Group histogram", fontdict={"fontsize": 18})
# rotate x labels
plt.ylabel("Count", fontdict={"fontsize": 18})
# add count on top of the bars
for i, v in enumerate(vip_group_counts):
    plt.text(i, v[1] + 0.15, str(v[1]), ha="center", va="bottom", fontsize=14)

# add horizontal grid lines
plt.grid(axis="y", zorder=0)
# ylim
plt.ylim(0, 30)
plt.tight_layout(pad=0)
plt.savefig("tasks/orbit_video_object_recognition/data/orbit_video_group_histogram.pdf")


# Make a wordcloud

from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud(
    width=600,
    height=600,
    background_color="white",
    collocations=False,
    repeat=False,
    max_words=50,
    stopwords=set(STOPWORDS),
    min_font_size=10,
).generate(",".join(vip_objects))
plt.figure(figsize=(6, 6), facecolor=None)
# plt.subplot(1, 2, 1)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("tasks/orbit_video_object_recognition/data/orbit_vip_object_wordcloud.pdf")

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
).generate(",".join(non_vip_objects))
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("tasks/orbit_video_object_recognition/data/orbit_object_wordcloud.pdf")


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
durations = [d for d in durations if d < 80]
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
plt.savefig("tasks/orbit_video_object_recognition/data/orbit_video_duration_histogram.pdf")
