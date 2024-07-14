import cv2
from ultralytics import YOLO
import os
import logging
import time
import subprocess
import boto3
from botocore.exceptions import NoCredentialsError

# Suppress ultralytics logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)

def download_file_from_s3(bucket_name, s3_key, local_path):
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name, s3_key, local_path)
    except NoCredentialsError:
        print("Credentials not available")

def upload_file_to_s3(bucket_name, local_path, s3_key):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_path, bucket_name, s3_key)
    except NoCredentialsError:
        print("Credentials not available")

def detect_and_save_video(video_path, output_path, model_path, target_classes, confidence_threshold=0.3):
    # Load the YOLOv8 model with GPU support
    model = YOLO(model_path).to('cuda')

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if not cap.isOpened():
        print("Error opening video file")
        return

    temp_output_video_path = os.path.join(output_path, "temp_output_video.mp4")
    final_output_video_path = os.path.join(output_path, "output_video.ts")
    video_writer = cv2.VideoWriter(temp_output_video_path, fourcc, fps, (width, height))

    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model.predict(frame)

        # Draw bounding boxes on the frame
        for result in results[0].boxes:
            cls = int(result.cls[0])
            conf = result.conf[0]  # Confidence score
            if model.names[cls] in target_classes and conf >= confidence_threshold:
                box = result.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[cls]}: {conf:.2f}"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label background
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(y1, label_size[1] + 10)
                cv2.rectangle(frame, (x1, label_ymin - label_size[1] - 10), (x1 + label_size[0], label_ymin + base_line - 10), (0, 255, 0), cv2.FILLED)

                # Draw label text
                cv2.putText(frame, label, (x1, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Write frame to the output video
        video_writer.write(frame)

    # Release resources
    video_writer.release()
    cap.release()

    # Convert MP4 to TS using ffmpeg
    convert_command = [
        'ffmpeg',
        '-i', temp_output_video_path,
        '-c:v', 'libx264',  # 비디오 코덱을 H.264로 설정
        '-c:a', 'copy',     # 오디오 스트림을 그대로 복사
        '-f', 'mpegts',     # 출력 포맷을 MPEG-TS로 지정
        final_output_video_path
    ]
    subprocess.run(convert_command, check=True)
    
    # Remove the temporary MP4 file
    os.remove(temp_output_video_path)
    
    print("Processing complete. Video saved as:", final_output_video_path)
    end_time = time.time()
    
    print(f"Time to process and save video: {end_time - start_time:.2f} seconds")

# S3 버킷 정보
bucket_name = 'boda-ts-bucket'
input_s3_key = 'test/test-14.ts'
output_s3_key = 'output/output.ts'

# 로컬 경로
local_input_path = '/tmp/input_video.ts'
local_output_path = '/tmp'

# 모델 경로 및 타겟 클래스 정의
model_path = 'v4_nano_results.pt'
target_classes = ['Person', 'Fallperson', 'Walkwithphone', 'Safetyhat']

# S3에서 파일 다운로드
start_time = time.time()
download_file_from_s3(bucket_name, input_s3_key, local_input_path)

# Ensure the output directory exists
os.makedirs(local_output_path, exist_ok=True)

# Run the function with a confidence threshold of 0.5
detect_and_save_video(local_input_path, local_output_path, model_path, target_classes, confidence_threshold=0.5)

# 결과 파일 S3에 업로드
upload_file_to_s3(bucket_name, os.path.join(local_output_path, "output_video.ts"), output_s3_key)
end_time = time.time()
print(f"Time to process and save video: {end_time - start_time:.2f} seconds")
