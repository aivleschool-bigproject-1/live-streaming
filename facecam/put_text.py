import cv2

class PutText:
    def __init__(self):
        self.red = (50, 50, 255)
    # 알파 블렌딩 함수
    def alpha_blend(self, frame_section, icon):
        alpha_s = icon[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            frame_section[:, :, c] = (alpha_s * icon[:, :, c] + alpha_l * frame_section[:, :, c])
        return frame_section

    # 아이콘과 텍스트 오버레이 함수
    def overlay_icon_and_text(self, frame, icon, text, x, y):
        icon_height, icon_width = icon.shape[:2]
        frame_section = frame[y:y+icon_height, x:x+icon_width]
        frame_section = self.alpha_blend(frame_section, icon)
        frame[y:y+icon_height, x:x+icon_width] = frame_section

        # 텍스트 위치 계산 (아이콘 중앙에 맞춤)
        text_x = x + icon_width + 10
        text_y = y + icon_height // 2 + 10
        if text == "Bad Posture":
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.red, 2)
        else:
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def display_log_on_frame(self, frame, posture_text, stress_index, bpm):
    
        # 아이콘 파일 경로
        heart_icon_path = "./heart.png"
        stress_icon_path = "./stress.png"
        user_icon_path = "./user.png"

        # 아이콘 이미지 로드
        heart_icon = cv2.imread(heart_icon_path, cv2.IMREAD_UNCHANGED)
        stress_icon = cv2.imread(stress_icon_path, cv2.IMREAD_UNCHANGED)
        user_icon = cv2.imread(user_icon_path, cv2.IMREAD_UNCHANGED)
    
        bpm_text = f'BPM: {bpm:.2f}'
        stress_text = f'Stress: {stress_index:.2f}'
        posture_text = posture_text
    
        # 프레임 크기 및 아이콘 배치 위치 정의
        frame_height, frame_width = 720, 1280
        icon_y_start = frame_height - 150

        # 아이콘과 텍스트 오버레이
        self.overlay_icon_and_text(frame, heart_icon, bpm_text, frame_width - 250, icon_y_start)
        self.overlay_icon_and_text(frame, stress_icon, stress_text, frame_width - 250, icon_y_start + 40)
        self.overlay_icon_and_text(frame, user_icon, posture_text, frame_width - 250, icon_y_start + 80)

        return frame