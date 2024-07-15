import cv2
import math as m
import mediapipe as mp

# Mediapipe 포즈 클래스 초기화
mp_pose = mp.solutions.pose

# 거리 계산
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# 각도 계산
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi * theta)
    return degree

def sendWarning(x):
    pass

def initialize_pose_model():
    return mp_pose.Pose(
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def process_frame(image, pose):
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    keypoints = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lm = keypoints.pose_landmarks
    lmPose = mp_pose.PoseLandmark

    shoulder_diff = 0
    forward_head_position = 0
    posture_status = "Unknown"

    if lm:
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        nose_x = int(lm.landmark[lmPose.NOSE].x * w)
        nose_y = int(lm.landmark[lmPose.NOSE].y * h)

        shoulder_diff = abs(l_shldr_y - r_shldr_y)
        forward_head_position = abs(nose_y - (l_shldr_y + r_shldr_y) / 2)
        overlay = image.copy()
        alpha = 0.3
        cv2.ellipse(overlay, (335, 170), (60, 90), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, (0, 255, 255), -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, (255, 0, 255), -1)
        cv2.circle(image, (nose_x, nose_y), 7, (255, 127, 0), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        angle_text_string = 'Shoulder diff : ' + str(int(shoulder_diff)) + ' degrees'
        position_text_string = 'Forward Head Position : ' + str(int(forward_head_position)) + ' px'

        if shoulder_diff < 50 and forward_head_position > 180:
            posture_status = "Good"
            cv2.putText(image, angle_text_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (127, 233, 100), 2)
            cv2.putText(image, position_text_string, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (127, 233, 100), 2)
            cv2.putText(image, 'Good Posture', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (127, 233, 100), 2)
            cv2.line(image, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), (127, 255, 0), 4)
            cv2.line(image, (nose_x, nose_y), (int((l_shldr_x + r_shldr_x) / 2), int((l_shldr_y + r_shldr_y) / 2)), (127, 255, 0), 4)
        else:
            posture_status = "Bad"
            cv2.putText(image, angle_text_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)
            cv2.putText(image, position_text_string, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)
            cv2.putText(image, 'Bad Posture', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 255), 2)
            cv2.line(image, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), (50, 50, 255), 4)
            cv2.line(image, (nose_x, nose_y), (int((l_shldr_x + r_shldr_x) / 2), int((l_shldr_y + r_shldr_y) / 2)), (50, 50, 255), 4)

    return image, shoulder_diff, forward_head_position, posture_status
