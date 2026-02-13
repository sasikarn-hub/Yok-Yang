import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO

# --- Config ---
try:
    model = YOLO('yolov8n-pose.pt')
except Exception as e:
    st.error("ไม่พบไฟล์ yolov8n-pose.pt ใน GitHub")
    st.stop()

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

if 'exercise_mode' not in st.session_state:
    st.session_state['exercise_mode'] = "Standing Bicep Curl"

class FitnessProcessor(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.set_count = 0
        self.stage = "down"
        self.reps_per_set = 10
        self.feedback = "READY?"  # ข้อความคำแนะนำ
        self.color = (255, 255, 0) # สีข้อความ (เริ่มต้นสีฟ้า/เหลือง)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        mode = st.session_state.get('exercise_mode', "Standing Bicep Curl")
        results = model(img, verbose=False, conf=0.5)
        
        try:
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            
            # จุดร่างกาย (ใช้ซีกขวา)
            p_sh = keypoints[6][:2]  # ไหล่
            p_el = keypoints[8][:2]  # ศอก
            p_wr = keypoints[10][:2] # ข้อมือ
            p_hip = keypoints[12][:2] # เอว

            # เช็คว่าเห็นคนชัดไหม
            if keypoints[6][2] > 0.5 and keypoints[8][2] > 0.5:
                
                # 1. BICEP CURL
                if mode == "Standing Bicep Curl":
                    angle = calculate_angle(p_sh, p_el, p_wr)
                    sway = abs(p_sh[0] - p_hip[0])
                    
                    if sway > 40: 
                        self.feedback = "LOCK YOUR BACK!"
                        self.color = (0, 0, 255) # สีแดง
                    elif p_el[1] < p_sh[1]: 
                        self.feedback = "KEEP ELBOWS DOWN!"
                        self.color = (0, 0, 255)
                    else:
                        self.feedback = "PERFECT FORM"
                        self.color = (0, 255, 0) # สีเขียว

                    if angle > 160: self.stage = "down"
                    if angle < 30 and self.stage == "down":
                        self.stage = "up"
                        self.counter += 1

                # 2. UPRIGHT ROW
                elif mode == "Standing Upright Row":
                    if p_el[1] < p_sh[1] - 30:  
                        self.feedback = "LOWER YOUR ELBOWS!"
                        self.color = (0, 0, 255)
                    elif p_wr[0] < p_sh[0] - 50: 
                        self.feedback = "WIDEN YOUR GRIP!"
                        self.color = (0, 0, 255)
                    else:
                        self.feedback = "GOOD SQUEEZE"
                        self.color = (0, 255, 0)

                    if p_wr[1] > p_hip[1]: self.stage = "down"
                    if p_wr[1] < p_sh[1] + 50 and self.stage == "down": 
                        self.stage = "up"
                        self.counter += 1

                # 3. FRONT RAISE
                elif mode == "Standing Front Raise":
                    arm_angle = calculate_angle(p_el, p_sh, p_hip)
                    if arm_angle > 100: 
                        self.feedback = "STOP AT EYE LEVEL!"
                        self.color = (0, 0, 255)
                    elif p_sh[0] < p_hip[0] - 30: 
                        self.feedback = "STAND STRAIGHT!"
                        self.color = (0, 0, 255)
                    else:
                        self.feedback = "NICE CONTROL"
                        self.color = (0, 255, 0)

                    if arm_angle < 20: self.stage = "down"
                    if arm_angle > 80 and arm_angle < 100 and self.stage == "down":
                        self.stage = "up"
                        self.counter += 1

                # จัดการ Set
                if self.counter >= self.reps_per_set:
                    self.set_count += 1
                    self.counter = 0

                # DRAW UI
                cv2.rectangle(img, (0, 0), (640, 60), self.color, -1)
                cv2.putText(img, self.feedback, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.rectangle(img, (0, 400), (200, 480), (0, 0, 0), -1)
                cv2.putText(img, f"REPS: {self.counter}", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, f"SETS: {self.set_count}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception:
            pass
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🏋️ AI Coach: Real-time Feedback")

option = st.selectbox(
    'เลือกท่าออกกำลังกาย:',
    ('Standing Bicep Curl', 'Standing Upright Row', 'Standing Front Raise')
)
st.session_state['exercise_mode'] = option

st.write("---")
st.write("**วิธีอ่านค่าบนหน้าจอ:**")
st.markdown("- 🟩 **สีเขียว (PERFECT FORM):** ทำดีแล้ว ทำต่อไป!")
st.markdown("- 🟥 **สีแดง (คำเตือน):** มีข้อผิดพลาด ให้ทำตามคำสั่งที่ขึ้นบนจอทันที")

webrtc_streamer(
    key="fitness-coach",
    video_processor_factory=FitnessProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}
)
