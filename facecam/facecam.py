# main.py
import os
import time
import datetime
import subprocess
import cv2
import numpy as np
import pytz
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.auth import HTTPBasicAuth
from collections import deque
from heart_rate_monitor import HeartRateMonitor
from posture_monitor import PostureMonitor
import tensorflow as tf

# TensorFlow 및 CUDA 경고 메시지 억제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 억제 수준 설정: 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# GPU 설정
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # print(f"Using GPU: {physical_devices}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU")

TEMP_OUTPUT_PATH = os.path.expanduser('~/codes/ts-processed')
MAX_BULK_SIZE = 90 * 1024 * 1024  # 90MB
ES_INDEX = 'facecam'

def get_timestamp():
    kst = pytz.timezone('Asia/Seoul')
    return datetime.datetime.now(kst).strftime('%Y%m%d%H%M%S')

def predict_heart_rate(frame, heart_rate_monitor):
    heart_rate_monitor.run(frame)
    return heart_rate_monitor.get_heart_rate()

def predict_posture(frame, posture_monitor):
    result = posture_monitor.run(frame)
    return result

def draw_results(frame, posture_result, heart_rate_monitor, graph_width, graph_height):
    frame_height, frame_width, _ = frame.shape
    output_width = frame_width + graph_width
    output_height = frame_height
    output_frame = np.ones((output_height, output_width, 3), dtype=np.uint8) * 0

    output_frame[:, :frame_width] = posture_result
    bpm_graph = heart_rate_monitor.plot_bpm(graph_width, graph_height)
    output_frame[:, frame_width:frame_width + graph_width] = bpm_graph
    
    return output_frame

def detect_and_save_video(video_path, output_path, tmp_path, heart_rate_monitor, posture_monitor, config):
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

    temp_output_video_path = os.path.join(tmp_path, f"{config['config_name']}_temp_video.mp4")
    timestamp = get_timestamp()
    final_output_video_path = os.path.join(output_path, f'{timestamp}-processed.ts')
    video_writer = cv2.VideoWriter(temp_output_video_path, fourcc, fps, (width + width, height))

    video_start_time = get_file_creation_time(video_path)

    frame_count = 0
    json_frame_count = 1
    bulk_data = ''

    previous_outputs = None
    previous_posture_result = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 2 == 0:
                future_heart_rate = executor.submit(predict_heart_rate, frame, heart_rate_monitor)
                future_posture = executor.submit(predict_posture, frame, posture_monitor)

                heart_rate = future_heart_rate.result()
                posture_result = future_posture.result()

                graph_width = width
                graph_height = height
                output_frame = draw_results(frame, posture_result, heart_rate_monitor, graph_width, graph_height)

                frame_timestamp = video_start_time + datetime.timedelta(seconds=(frame_count / fps))
                detection_info = {
                    'frame': json_frame_count,
                    'timestamp': frame_timestamp.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    'heart_rate': heart_rate,
                    'posture_status': posture_monitor.get_posture_status()
                }

                bulk_data += json.dumps({"index": {"_index": ES_INDEX}}) + '\n'
                bulk_data += json.dumps(detection_info) + '\n'

                if len(bulk_data.encode('utf-8')) > MAX_BULK_SIZE:
                    send_bulk_to_elasticsearch(bulk_data, ES_INDEX)
                    bulk_data = ''

                previous_outputs = heart_rate
                previous_posture_result = posture_result
                json_frame_count += 1
            else:
                if previous_outputs is not None and previous_posture_result is not None:
                    output_frame = draw_results(frame, previous_posture_result, heart_rate_monitor, width, height)
                else:
                    output_frame = frame

            video_writer.write(output_frame)

            frame_count += 1

    if bulk_data:
        send_bulk_to_elasticsearch(bulk_data, ES_INDEX)

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

def send_bulk_to_elasticsearch(bulk_data, es_index):
    url = f'https://ea86ace4539b432f84fc0f19c4c0c586.ap-northeast-2.aws.elastic-cloud.com/{es_index}/_bulk'
    headers = {'Content-Type': 'application/x-ndjson'}
    username = 'elastic'
    password = 'B8p8BRDeQ0FTCcEJFdR6Bf6U'

    try:
        response = requests.post(url, headers=headers, data=bulk_data, auth=HTTPBasicAuth(username, password))
        # print("response:", response.status_code, response.text)
    except requests.exceptions.SSLError as e:
        print("SSL error:", e)
    except requests.exceptions.ConnectionError as e:
        print("Error connecting to Elasticsearch:", e)
    except Exception as e:
        print("An error occurred:", e)

def main(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    process_ts_path = os.path.expanduser(config['process_ts_path'])
    processed_ts_path = os.path.expanduser(config['processed_ts_path'])
    m3u8_filename = os.path.join(processed_ts_path, 'playlist.m3u8')

    config['config_name'] = os.path.splitext(os.path.basename(config_path))[0]

    heart_rate_monitor = HeartRateMonitor()
    posture_monitor = PostureMonitor()

    while True:
        next_file = get_next_file_to_process(process_ts_path)
        if next_file:
            video_path = os.path.join(process_ts_path, next_file)
            video_start_time = time.time()
            detect_and_save_video(video_path, processed_ts_path, TEMP_OUTPUT_PATH, heart_rate_monitor, posture_monitor, config)
            video_end_time = time.time()
            os.remove(video_path)

            maintain_max_files(processed_ts_path, 30)
            update_m3u8(processed_ts_path, m3u8_filename)
            print(f"Total processing time for {video_path}: {video_end_time - video_start_time:.4f} seconds")
        else:
            time.sleep(5)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CCTV Video Processing')
    parser.add_argument('config', type=str, help='Path to the config file')
    args = parser.parse_args()
    main(args.config)
