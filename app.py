import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO

# 1. โหลดโมเดล (ต้องมั่นใจว่ามีไฟล์ .pt ใน GitHub นะครับ)
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n-pose.pt')

try:
    model = load_yolo()
except Exception as e:
    st.error("ไม่พบไฟล์ yolov8n-pose.pt ใน GitHub กรุณาตรวจสอบการอัปโหลดครับ")
    st.stop()

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

class FitnessProcessor(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.stage = "down"
        self.feedback = "READY?"
        self.color = (0, 255, 255)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        mode = st.session_state.get('exercise_mode', "Standing Bicep Curl")
        
        results = model(img, verbose=False, conf=0.5)
        
        try:
            if results[0].keypoints.data.shape[1] > 0:
                kpts = results[0].keypoints.data[0].cpu().numpy()
                # ไหล่(6), ศอก(8), ข้อมือ(10), เอว(12)
                p_sh, p_el, p_wr, p_hp = kpts[6][:2], kpts[8][:2], kpts[10][:2], kpts[12][:2]

                if kpts[6][2] > 0.5:
                    if mode == "Standing Bicep Curl":
                        ang = calculate_angle(p_sh, p_el, p_wr)
                        sway = abs(p_sh[0] - p_hp[0])
                        if sway > 35:
                            self.feedback, self.color = "LOCK YOUR BACK!", (0, 0, 255)
                        else:
                            self.feedback, self.color = "GOOD FORM", (0, 255, 0)
                        if ang > 160: self.stage = "down"
                        if ang < 30 and self.stage == "down" and sway <= 35:
                            self.stage, self.counter = "up", self.counter + 1
                    
                    elif mode == "Standing Upright Row":
                        if p_el[1] < p_sh[1] - 20:
                            self.feedback, self.color = "LOWER ELBOWS!", (0, 0, 255)
                        else:
                            self.feedback, self.color = "NICE SQUEEZE", (0, 255, 0)
                        if p_wr[1] > p_hp[1]: self.stage = "down"
                        if p_wr[1] < p_sh[1] + 40 and self.stage == "down":
                            self.stage, self.counter = "up", self.counter + 1

                    elif mode == "Standing Front Raise":
                        ang = calculate_angle(p_el, p_sh, p_hp)
                        if ang > 95:
                            self.feedback, self.color = "STOP AT SHOULDER!", (0, 0, 255)
                        else:
                            self.feedback, self.color = "CONTROLLED", (0, 255, 0)
                        if ang < 20: self.stage = "down"
                        if ang > 80 and self.stage == "down" and ang <= 95:
                            self.stage, self.counter = "up", self.counter + 1

                # วาด UI ลงในวิดีโอ
                cv2.rectangle(img, (0, 0), (640, 65), self.color, -1)
                cv2.putText(img, f"{self.feedback} | REPS: {self.counter}", (15, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        except:
            pass
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI ส่วนหน้าเว็บ
st.set_page_config(page_title="Coach Krob", layout="centered")
st.title("🏋️ Coach Krob: AI Trainer")
option = st.selectbox('เลือกท่าออกกำลังกาย:', ('Standing Bicep Curl', 'Standing Upright Row', 'Standing Front Raise'))
st.session_state['exercise_mode'] = option

# เรียกใช้ระบบกล้อง
webrtc_streamer(
    key="coach-krob-final",
    video_processor_factory=FitnessProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
