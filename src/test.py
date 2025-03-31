import os
import cv2
import subprocess
import numpy as np
from google.cloud import storage
from shutil import rmtree
import time
import random
from tqdm import tqdm
import io
from PIL import Image
import json


# Initialize GCP storage client
client = storage.Client()
bucket_name = 'physical-social-norm'
bucket = client.get_bucket(bucket_name)

# Get list of already uploaded videos
blobs = bucket.list_blobs()
uploaded_videos1 = [i.name.split('/')[-1].split('_')[0] + '_' + i.name.split('/')[-1].split('_')[1] for i in blobs if i.name.startswith('sampled_snippets_new_new/') and len(i.name.split('/')[-1]) > 1]
print(f"{len(uploaded_videos1)} videos already uploaded to GCP1")
blobs = bucket.list_blobs()
uploaded_videos2 = [i.name.split('/')[-1].split('_')[0] + '_' + i.name.split('/')[-1].split('_')[1] for i in blobs if i.name.startswith('sampled_snippets_v2/') and len(i.name.split('/')[-1]) > 1]
print(f"{len(uploaded_videos2)} videos already uploaded to GCP2")
blobs = bucket.list_blobs()
all_uploaded_videos = {i.name.split('/')[-2] for i in blobs if i.name.endswith('frame_all_random.jpg') and len(i.name.split('/')[-1]) > 1}
print(f"{len(all_uploaded_videos)} videos already uploaded to GCP")

data = list(json.load(open('./final_dataset/final_data.json', 'r')).keys())

if os.path.exists('./frames'):
    rmtree('./frames')


os.makedirs('./frames', exist_ok=True)

def create_blank_image(width, height, color=(255, 255, 255)):
    return np.full((height, width, 3), color, dtype=np.uint8)

def download_image(bucket, blob_path):
    blob = bucket.blob(blob_path)
    try:
        # Download image to memory
        image_bytes = blob.download_as_bytes()
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error downloading {blob_path}: {e}")
        return None

for i, vid in tqdm(enumerate(data), total=len(data)):
    if vid in all_uploaded_videos:
        continue
    if vid in uploaded_videos1:
        mode = 'new_new'
    elif vid in uploaded_videos2:
        mode = 'v2'

    output_frame_dir = f'./frames/{vid}'
    os.makedirs(output_frame_dir, exist_ok=True)

    # Initialize variables
    cols = 5
    rows = 1
    frame_images = []
    
    # Download all available frames (0-4)
    for frame_idx in range(5):
        blob_path = f'sampled_frames_{mode}/{vid}/frame_{frame_idx}_prev.jpg'
        image = download_image(bucket, blob_path)
        
        if image is not None:
            frame_images.append(image)
        
    # If no images were found, skip this video
    if not frame_images:
        print(f"No images found for video {vid}, skipping...")
        continue
    
    # Get dimensions from the first frame
    frame_height, frame_width = frame_images[0].shape[:2]
    
    # Create a blank combined image
    combined_image = create_blank_image(cols * frame_width, rows * frame_height)
    
    # Shuffle the frame images to randomize order
    random.shuffle(frame_images)
    
    # Fill in available frames
    for frame_index, image in enumerate(frame_images):
        if frame_index >= cols:  # Only use up to 5 frames
            break
            
        col_idx = frame_index % cols
        combined_image[0:frame_height, col_idx * frame_width:(col_idx + 1) * frame_width] = image
    
    # Save the combined image
    if combined_image is not None:
        combined_filename = os.path.join(output_frame_dir, f'frame_all_random.jpg')
        cv2.imwrite(combined_filename, combined_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
    # Upload every 100 videos to avoid memory issues
    if (i + 1) % 100 == 0:
        time.sleep(10)
        print(f'Uploading Frames')
        upload_command = f"bash upload.sh"
        subprocess.run(upload_command, shell=True)

        print(f'Removing Frames')
        if os.path.exists('./frames'):
            rmtree('./frames')
            os.makedirs('./frames', exist_ok=True)

# Final upload
time.sleep(10)
print(f'Uploading Frames')
upload_command = f"bash upload.sh"
subprocess.run(upload_command, shell=True)

print(f'Removing Frames')
if os.path.exists('./frames'):
    rmtree('./frames')

print("Process completed.")