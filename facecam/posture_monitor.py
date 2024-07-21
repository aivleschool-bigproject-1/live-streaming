import cv2
import mediapipe as mp
import math as m
import tensorflow as tf

class PostureMonitor:
    def __init__(self):
        self.good_frames = 0
        self.bad_frames = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.blue = (255, 127, 0)
        self.red = (50, 50, 255)
        self.green = (127, 255, 0)
        self.light_green = (127, 233, 100)
        self.yellow = (0, 255, 255)
        self.pink = (255, 0, 255)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # GPU 설정
        self.device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

    def findDistance(self, x1, y1, x2, y2):
        return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def findAngle(self, x1, y1, x2, y2):
        theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
        return int(180 / m.pi * theta)

    def run(self, image):
        with tf.device(self.device):
            h, w = image.shape[:2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            keypoints = self.pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            lm = keypoints.pose_landmarks
            lmPose = self.mp_pose.PoseLandmark

            if lm:
                l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
                l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
                r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
                r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
                nose_x = int(lm.landmark[lmPose.NOSE].x * w)
                nose_y = int(lm.landmark[lmPose.NOSE].y * h)

                shoulder_diff = abs(l_shldr_y - r_shldr_y)
                forward_head_position = abs(nose_y - (l_shldr_y + r_shldr_y) / 2)

                if shoulder_diff < 50 and forward_head_position > 95:
                    self.bad_frames = 0
                    self.good_frames += 1
                    cv2.line(image, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), self.green, 4)
                    cv2.line(image, (nose_x, nose_y), (int((l_shldr_x + r_shldr_x) / 2), int((l_shldr_y + r_shldr_y) / 2)), self.green, 4)

                else:
                    self.good_frames = 0
                    self.bad_frames += 1
                    cv2.line(image, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), self.red, 4)
                    cv2.line(image, (nose_x, nose_y), (int((l_shldr_x + r_shldr_x) / 2), int((l_shldr_y + r_shldr_y) / 2)), self.red, 4)

            return image

    def sendWarning(self):
        pass

    def get_posture_status(self):
        if self.good_frames > self.bad_frames:
            return 'Good'
        else:
            return 'Bad'
