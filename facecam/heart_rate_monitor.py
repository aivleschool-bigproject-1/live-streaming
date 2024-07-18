import cv2
import dlib
import numpy as np
import time
from scipy.signal import butter, lfilter
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import math
import tensorflow as tf

class HeartRateMonitor:
    def __init__(self, buffer_size=150):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=buffer_size)
        self.times = deque(maxlen=buffer_size)
        self.bpm = 0
        self.t0 = time.time()
        self.initialized = False  # 버퍼 초기화 상태 확인

        # 스트레스 지수 계산용 데이터
        self.bpm_values_count = 0
        self.sum_squared_differences = 0.0
        self.previous_bpm_value = None

        # GPU 설정
        self.device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

    def initialize_buffers(self, initial_value):
        np.random.seed(42)  # 임의 데이터 생성의 재현성을 위해 시드 설정
        noise = np.random.normal(0, 2, self.buffer_size)  # 약간의 노이즈 추가
        for i in range(self.buffer_size):
            t = i / 30.0  # 30 FPS 가정
            green_mean = initial_value + noise[i]  # 초기 값에 노이즈 추가
            self.data_buffer.append(green_mean)
            self.times.append(t)
        self.initialized = True  # 버퍼 초기화 완료 표시

    def get_landmarks(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) > 0:
            landmarks = self.predictor(gray, faces[0])
            return [(p.x, p.y) for p in landmarks.parts()], faces[0]
        return None, None

    def get_forehead_roi(self, landmarks, image):
        if landmarks:
            x1, y1 = landmarks[21]
            x2, y2 = landmarks[22]
            forehead = image[y1-15:y1+15, x1-15:x2+15]
            return forehead
        return None

    def extract_green_channel_mean(self, roi):
        if roi is not None and roi.size > 0:
            green_channel = roi[:, :, 1]
            return np.mean(green_channel)
        return None

    def run(self, frame):
        with tf.device(self.device):
            # print("CUDA 지원 여부: ", dlib.DLIB_USE_CUDA)
            landmarks, face = self.get_landmarks(frame)
            if face is not None:
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 2)
            roi = self.get_forehead_roi(landmarks, frame)
            green_mean = self.extract_green_channel_mean(roi)

            if green_mean is not None:
                for i in range(5):      # 버퍼와 시간 스킵한 프레임수만큼 추가
                    self.data_buffer.append(green_mean)
                    self.times.append(time.time() - self.t0)
                if len(self.data_buffer) >= self.buffer_size:
                    self.compute_heart_rate()

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        if fs <= 0:
            raise ValueError("Sampling frequency must be positive and non-zero.")
    
        nyquist = 0.5 * fs

        if (highcut >= nyquist):
            nyquist += (math.ceil((highcut-nyquist) * 1000) / 1000)
        
        low = lowcut / nyquist
        high = highcut / nyquist
        
        if low <= 0 or high >= 1:
            raise ValueError("Digital filter critical frequencies must be 0 < Wn < 1.")
    
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y

    def compute_heart_rate(self):
        if len(self.data_buffer) < 2:
            return
    
        fs = len(self.data_buffer) / (self.times[-1] - self.times[0])
        if fs <= 0:
            print("Sampling frequency is non-positive, skipping heart rate calculation.")
            return

        try:
            filtered = self.butter_bandpass_filter(self.data_buffer, 0.8, 3.5, fs, order=5)
        except ValueError as e:
            print(f"Error in filtering: {e}")
            return

        fft = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(filtered), 1.0/fs)
        idx = np.argmax(fft)
        self.bpm = freqs[idx] * 60.0
        self.bpm_values_count += 1

    def calculate_stress_index(self):
        # 심박수 2개부터 계산
        if self.bpm_values_count < 2:
            self.previous_bpm_value = self.bpm
            return 0.0
        
        difference = self.bpm - self.previous_bpm_value
        self.sum_squared_differences += difference * difference
        self.previous_bpm_value = self.bpm

        rmssd = math.sqrt(self.sum_squared_differences / (self.bpm_values_count - 1))
        stress_index = (rmssd / 50) * 100

        # 심박수 데이터 최소 300개일때부터 계산
        if self.bpm_values_count < 2:
            return 0.0  # 충분한 데이터가 없음
        else:
            return max(0, min(100, stress_index))

    def get_stress(self):
        stress_index = self.calculate_stress_index()
        return stress_index
    
    def get_heart_rate(self):
        return self.bpm

    def display_bpm_on_frame(self, frame):
        bpm_text = f'BPM: {self.bpm:.2f}'
        text_size = cv2.getTextSize(bpm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = frame.shape[1] - text_size[0] - 10
        text_y = 30
        cv2.putText(frame, bpm_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        stress_index = self.calculate_stress_index()
        stress_text = f'Stress: {stress_index:.2f}'
        stress_text_size = cv2.getTextSize(stress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        stress_text_x = frame.shape[1] - stress_text_size[0] - 10
        stress_text_y = text_y + 30
        cv2.putText(frame, stress_text, (stress_text_x, stress_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame
