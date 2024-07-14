import os
import time
import subprocess
import cv2
from ultralytics import YOLO
import m3u8

PROCESS_TS_PATH = os.path.expanduser('~/codes/s3bucket/industrial-cctv-input')
PROCESSED_TS_PATH = os.path.expanduser('~/codes/s3bucket/industrial-cctv-output')
TEMP_OUTPUT_PATH = os.path.expanduser('~/codes/ts-processed')
M3U8_FILENAME = os.path.join(PROCESSED_TS_PATH, 'playlist.m3u8')
MODEL_PATHS = ['best.pt', 'v4_nano_results.pt']
TARGET_CLASSES = ['Person', 'Fallperson', 'Walkwithphone', 'Safetyhat', 'Fire']
CONFIDENCE_THRESHOLD = 0.3

def detect_and_save_video(video_path, output_path, tmp_path, model_paths, target_classes, confidence_threshold=0.3):
    models = [YOLO(model_path).to('cuda') for model_path in model_paths]
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if not cap.isOpened():
        print("Error opening video file")
        return

    temp_output_video_path = os.path.join(tmp_path, "temp_output_video.mp4")
    final_output_video_path = os.path.join(output_path, os.path.basename(video_path).replace('.ts', '-processed.ts'))
    video_writer = cv2.VideoWriter(temp_output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        for model in models:
            results = model.predict(frame)

            for result in results[0].boxes:
                cls = int(result.cls[0])
                conf = result.conf[0]
                if model.names[cls] in target_classes and conf >= confidence_threshold:
                    box = result.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{model.names[cls]}: {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_ymin = max(y1, label_size[1] + 10)
                    cv2.rectangle(frame, (x1, label_ymin - label_size[1] - 10), (x1 + label_size[0], label_ymin + base_line - 10), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, label, (x1, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        video_writer.write(frame)

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
    print("Processing complete. Video saved as:", final_output_video_path)

def get_next_file_to_process(process_path):
    files = [f for f in os.listdir(process_path) if f.endswith('.ts')]
    files.sort(key=lambda f: int(f.split('-')[1].split('.')[0]))
    return files[0] if files else None

def maintain_max_files(directory, max_files):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.ts')]
    files.sort(key=os.path.getctime)
    while len(files) > max_files:
        os.remove(files.pop(0))
        print(f"Deleted old file to maintain max files: {files[0]}")

def update_m3u8(directory, m3u8_filename):
     # Get all .ts files in the directory
    ts_files = [f for f in os.listdir(directory) if f.endswith('.ts')]
    
    # Sort files by creation time
    ts_files = sorted(ts_files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    
    # Header of the M3U8 file
    playlist_header = [
        '#EXTM3U',
        '#EXT-X-VERSION:3',
        '#EXT-X-TARGETDURATION:10',
        '#EXT-X-MEDIA-SEQUENCE:0'
    ]
    
    # Segments information
    playlist_segments = []
    for ts_file in ts_files:
        playlist_segments.append(f'#EXTINF:10,')
        playlist_segments.append(ts_file)
    
    # End of the M3U8 file
    playlist_footer = ['#EXT-X-ENDLIST']
    
    # Combine all parts of the playlist
    playlist_content = '\n'.join(playlist_header + playlist_segments + playlist_footer)
    
    # Write to the M3U8 file
    with open(m3u8_filename, 'w') as f:
        f.write(playlist_content)
    print(f"M3U8 file updated: {m3u8_filename}")

def main():
    while True:
        next_file = get_next_file_to_process(PROCESS_TS_PATH)
        if next_file:
            video_path = os.path.join(PROCESS_TS_PATH, next_file)
            detect_and_save_video(video_path, PROCESSED_TS_PATH, TEMP_OUTPUT_PATH, MODEL_PATHS, TARGET_CLASSES, CONFIDENCE_THRESHOLD)
            os.remove(video_path)  # Remove the processed file

            maintain_max_files(PROCESSED_TS_PATH, 30)
            update_m3u8(PROCESSED_TS_PATH, M3U8_FILENAME)
        else:
            time.sleep(5)  # Wait for a short while before checking again

if __name__ == "__main__":
    main()