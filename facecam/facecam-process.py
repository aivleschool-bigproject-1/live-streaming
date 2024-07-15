import os
import time
import subprocess
import cv2
import datetime
import json
import argparse
import pytz
from posture_detection import initialize_pose_model, process_frame

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

TEMP_OUTPUT_PATH = os.path.expanduser('~/codes/ts-processed')
CONFIDENCE_THRESHOLD = 0.3

def get_timestamp():
    kst = pytz.timezone('Asia/Seoul')
    return datetime.datetime.now(kst).strftime('%Y%m%d%H%M%S')

def detect_and_save_video(video_path, output_path, tmp_path, config_name):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = total_frames / fps if fps > 0 else 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if not cap.isOpened():
        print("Error opening video file")
        return

    temp_output_video_path = os.path.join(tmp_path, f"{config_name}_temp_video.mp4")
    timestamp = get_timestamp()
    final_output_video_path = os.path.join(output_path, f'{timestamp}-processed.ts')
    video_writer = cv2.VideoWriter(temp_output_video_path, fourcc, fps, (width, height))

    json_output_path = os.path.join(tmp_path, f'{timestamp}-detection-counts.json')
    detection_counts = []

    video_start_time = get_file_creation_time(video_path)

    pose_model = initialize_pose_model()

    frame_count = 0
    json_frame_count = 1
    previous_processed_frame = None
    previous_shoulder_diff = None
    previous_forward_head_position = None
    previous_posture_status = None

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 2 == 0:
            processed_frame, shoulder_diff, forward_head_position, posture_status = process_frame(frame, pose_model)
            frame_timestamp = video_start_time + datetime.timedelta(seconds=(frame_count / fps))
            detection_counts.append({
                'frame': json_frame_count,
                'timestamp': frame_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'shoulder_diff': shoulder_diff,
                'forward_head_position': forward_head_position,
                'posture_status': posture_status
            })
            json_frame_count += 1

            previous_processed_frame = processed_frame
            previous_shoulder_diff = shoulder_diff
            previous_forward_head_position = forward_head_position
            previous_posture_status = posture_status
        else:
            processed_frame = previous_processed_frame
            shoulder_diff = previous_shoulder_diff
            forward_head_position = previous_forward_head_position
            posture_status = previous_posture_status

        video_writer.write(processed_frame)
        frame_count += 1

    end_time = time.time()

    with open(json_output_path, 'w') as json_file:
        json.dump(detection_counts, json_file, indent=4)

    video_writer.release()
    cap.release()

    convert_command = [
        'ffmpeg',
        '-i', temp_output_video_path,
        '-c:v', 'libx264',
        '-c:a', 'copy',
        '-f', 'mpegts',
        final_output_video_path
    ]
    subprocess.run(convert_command, check=True)
    os.remove(temp_output_video_path)

    # process_duration = end_time - start_time
    # print(f"TS file video length: {video_length:.2f} seconds")
    # print(f"Processing duration: {process_duration:.2f} seconds")
    # print("Processing complete. Video saved as:", final_output_video_path)

def get_file_creation_time(file_path):
    creation_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
    return creation_time.astimezone(pytz.timezone('Asia/Seoul'))

def get_next_file_to_process(process_path):
    files = [f for f in os.listdir(process_path) if f.endswith('.ts')]
    files.sort(key=lambda f: int(f.split('-')[1].split('.')[0]))
    return files[0] if files else None

def maintain_max_files(directory, max_files):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.ts')]
    files.sort(key=os.path.getctime)
    while len(files) > max_files:
        os.remove(files.pop(0))

def update_m3u8(directory, m3u8_filename):
    ts_files = [f for f in os.listdir(directory) if f.endswith('.ts')]
    ts_files = sorted(ts_files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))

    playlist_header = [
        '#EXTM3U',
        '#EXT-X-VERSION:3',
        '#EXT-X-TARGETDURATION:10',
        '#EXT-X-MEDIA-SEQUENCE:0'
    ]

    playlist_segments = []
    for ts_file in ts_files:
        playlist_segments.append(f'#EXTINF:10,')
        playlist_segments.append(ts_file)

    playlist_footer = ['#EXT-X-ENDLIST']

    playlist_content = '\n'.join(playlist_header + playlist_segments + playlist_footer)

    with open(m3u8_filename, 'w') as f:
        f.write(playlist_content)

def main(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    process_ts_path = os.path.expanduser(config['process_ts_path'])
    processed_ts_path = os.path.expanduser(config['processed_ts_path'])
    m3u8_filename = os.path.join(processed_ts_path, 'playlist.m3u8')

    config_name = os.path.splitext(os.path.basename(config_path))[0]

    while True:
        next_file = get_next_file_to_process(process_ts_path)
        if next_file:
            video_path = os.path.join(process_ts_path, next_file)
            detect_and_save_video(video_path, processed_ts_path, TEMP_OUTPUT_PATH, config_name)
            os.remove(video_path)  # Remove the processed file

            maintain_max_files(processed_ts_path, 30)
            update_m3u8(processed_ts_path, m3u8_filename)
        else:
            time.sleep(5)  # Wait for a short while before checking again

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CCTV Video Processing')
    parser.add_argument('config', type=str, help='Path to the config file')
    args = parser.parse_args()
    main(args.config)
