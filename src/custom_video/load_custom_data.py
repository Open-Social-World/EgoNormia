import json
import cv2
import numpy as np
import tqdm
import os

video_file_paths = []

template = {
    "0": {
        "id": "0",
        "behaviors": [
            "",
            "",
            "",
            "",
            ""
        ],
        "justifications": [
            "",
            "",
            "",
            "",
            ""
        ],
        "correct": None,
        "sensibles": [],
        "taxonomy": {
            "0": [],
            "1": [],
            "2": [],
            "3": [],
            "4": []
        },
        "desc": ""
    }
}


# Iterate over all files in /raw, extract those that are videos
for file in os.listdir('raw'):
    if file.endswith('.mp4'):
        video_file_paths.append(file)

    # Sort by int in filename
    video_file_paths = sorted(video_file_paths, key=lambda x: int(x.split('.')[0]))

# if len(video_file_paths) == 0:
#     print("No videos found in custom_video/raw - exiting.")
#     exit()
# Load raw/custom_file.json
with open('raw/custom_file.json', 'r') as file:
    custom_data = json.load(file)
if 'id_goes_here' in custom_data.keys() or len(custom_data.keys()) != len(video_file_paths):
    print("Custom data file empty or insufficiently populated - creating empty json instead.")
    print("Process this with the pipeline in gen.")
    # Repeat custom_data for each video
    custom_data = {}
    for i in range(len(video_file_paths)):
        datapoint = template['0']
        datapoint['id'] = str(i)

        custom_data[str(i)] = datapoint
else:
    print("Custom data loaded successfully.")

for vid in tqdm.tqdm(custom_data.keys()):
    # Load video from /raw
    video = cv2.VideoCapture('raw/' + video_file_paths[int(vid)])
    # Get the frames per second
    fps = video.get(cv2.CAP_PROP_FPS)

    # Get first frame
    success, image = video.read()

    # Iterate over rest of frames, concat to image
    cnt = 1
    while success and cnt < 5:
        success, frame = video.read()
        # Only concact if every fps frame
        if cnt % fps == 0:
            image = np.concatenate((image, frame), axis=1)

    # Save the image
    cv2.imwrite(f'frames/{vid}.jpg', image)

    # Close the video
    video.release()

print("Video conversion complete.")

# Save custom_data to final_datset/custom_data.json
# Create final_dataset/custom_data_eval.json, which is just keys of custom_data.json

with open('../final_dataset/custom_data.json', 'w') as file:
    json.dump(custom_data, file)

custom_data_eval = {k:{} for k in custom_data.keys()}

with open('../final_dataset/custom_data_eval.json', 'w') as file:
    json.dump(custom_data_eval, file)





