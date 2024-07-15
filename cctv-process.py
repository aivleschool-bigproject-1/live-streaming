import os
import time
import subprocess
import cv2
from ultralytics import YOLO
import m3u8
import datetime
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
import pytz

TEMP_OUTPUT_PATH = os.path.expanduser('~/codes/ts-processed')
MODEL_PATHS = ['fire_seg_results.pt', 'v8_nano_results.pt']

def get_timestamp():
    kst = pytz.timezone('Asia/Seoul')
    return datetime.datetime.now(kst).strftime('%Y%m%d%H%M%S')

def load_model(model_path):
    model = YOLO(model_path).to('cuda')
    return {'model': model, 'path': model_path}

def predict_model(frame, model, target_classes, confidence_threshold):
    results = model['model'].predict(frame)
    output = []
    class_counts = {cls: 0 for cls in target_classes}
    segmentation_masks = []

    for result in results[0].boxes:
        cls = int(result.cls[0])
        conf = result.conf[0]
        class_name = model['model'].names[cls]
        if class_name in target_classes and conf >= confidence_threshold:
            box = result.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_name}: {conf:.2f}"
            output.append((x1, y1, x2, y2, label))
            class_counts[class_name] += 1
    
    if 'fire_seg_results.pt' in model['path']:
        if results[0].masks:
            segmentation_masks = results[0].masks.xy

    return output, class_counts, segmentation_masks

def draw_bounding_boxes(frame, outputs):
    for (x1, y1, x2, y2, label) in outputs:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(y1, label_size[1] + 10)
        cv2.rectangle(frame, (x1, label_ymin - label_size[1] - 10), (x1 + label_size[0], label_ymin + base_line - 10), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, label, (x1, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def draw_segmentation(frame, outputs, masks):
    draw_bounding_boxes(frame, outputs)
    for mask in masks:
        for m in mask:
            m = m.astype(int)
            frame[m[1], m[0]] = [255, 0, 255]

def detect_and_save_video(video_path, output_path, tmp_path, models, target_classes, config):
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
    video_writer = cv2.VideoWriter(temp_output_video_path, fourcc, fps, (width, height))

    json_output_path = os.path.join(tmp_path, f'{timestamp}-detection-counts.json')
    detection_counts = []

    video_start_time = get_file_creation_time(video_path)

    frame_count = 0
    json_frame_count = 1  # Initialize JSON frame count
    previous_outputs = []
    previous_class_counts = {cls: 0 for cls in target_classes}

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 2 == 0:  # Process every second frame
                futures = []
                for model in models:
                    if 'fire_seg_results.pt' in model['path']:
                        futures.append(executor.submit(predict_model, frame, model, target_classes, 0.3))
                    else:
                        futures.append(executor.submit(predict_model, frame, model, target_classes, 0.65))
                        
                frame_detections = []
                frame_class_counts = {cls: 0 for cls in target_classes}
                segmentation_masks = []
                for i, future in enumerate(futures):
                    outputs, class_counts, masks = future.result()
                    frame_detections.extend(outputs)
                    for cls, count in class_counts.items():
                        frame_class_counts[cls] += count
                    if masks:
                        segmentation_masks.extend(masks)
                previous_outputs = frame_detections
                previous_class_counts = frame_class_counts

                frame_timestamp = video_start_time + datetime.timedelta(seconds=(frame_count / fps))
                detection_info = {
                    'frame': json_frame_count,
                    'timestamp': frame_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'detections': frame_class_counts
                }
                if config['config_name'] == 'industrial_config':
                    detection_info['min_required_personnel'] = config['min_required_personnel']
                detection_counts.append(detection_info)
                json_frame_count += 1  # Increment JSON frame count
            else:
                frame_detections = previous_outputs
                frame_class_counts = previous_class_counts

            if frame_detections:
                if 'fire_seg_results.pt' in models[0]['path']:
                    draw_segmentation(frame, frame_detections, segmentation_masks)
                else:
                    draw_bounding_boxes(frame, frame_detections)

            video_writer.write(frame)
            frame_count += 1

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
    target_classes = config['target_classes']
    m3u8_filename = os.path.join(processed_ts_path, 'playlist.m3u8')

    config['config_name'] = os.path.splitext(os.path.basename(config_path))[0]

    models = [load_model(model_path) for model_path in MODEL_PATHS]

    while True:
        next_file = get_next_file_to_process(process_ts_path)
        if next_file:
            video_path = os.path.join(process_ts_path, next_file)
            detect_and_save_video(video_path, processed_ts_path, TEMP_OUTPUT_PATH, models, target_classes, config)
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
