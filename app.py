import cv2
import numpy as np
from ultralytics import YOLO

# --- Config & Colors ---
# สี (B, G, R)
COLOR_PRIMARY = (255, 191, 0)    # สีฟ้า Deep Sky Blue
COLOR_SECONDARY = (0, 255, 0)    # สีเขียว Lime
COLOR_WARNING = (0, 0, 255)      # สีแดง
COLOR_TEXT = (255, 255, 255)     # สีขาว
COLOR_BG = (20, 20, 20)          # สีเทาเข้มเกือบดำ

# โหลดโมเดล
print("Loading AI Model...")
model = YOLO('yolov8n-pose.pt') 

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def draw_rounded_rect(img, pt1, pt2, color, thickness, r, d):
    # ฟังก์ชันวาดกล่องแบบมีขอบมน (จำลอง) หรือใช้ Overlay ธรรมดาเพื่อความเร็ว
    x1, y1 = pt1
    x2, y2 = pt2
    # วาดสี่เหลี่ยมโปร่งใส
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img) # 0.6 คือความทึบ (ปรับลดได้)
    return img

# ตัวแปรระบบ
exercise = "LATERAL RAISE"
counter = 0
set_count = 0
stage = "ready"
score = 0
ideal = 0
window_name = "AI Fitness HUD Pro"

cap = cv2.VideoCapture(0)
# บังคับความละเอียด HD (เพื่อความสวยงาม)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # กลับด้านกระจก
    frame = cv2.flip(frame, 1)
    
    # YOLO Detect
    results = model(frame, verbose=False, conf=0.5)
    
    # สร้างภาพสำหรับวาด UI (แยกจากภาพ YOLO เดิมเพื่อให้สะอาดตา)
    # ถ้าอยากเห็นเส้นกระดูก ให้ใช้ frame = results[0].plot() แทนบรรทัดล่าง
    # frame = results[0].plot() 
    
    # แต่ถ้าอยากได้แบบ Clean (วาดเส้นเอง หรือไม่วาดเลย) ให้ใช้ frame เดิม
    # ในที่นี้ขอใช้ frame เดิม แล้ววาดเฉพาะจุดสำคัญ จะดู Pro กว่า
    
    try:
        keypoints = results[0].keypoints.data[0].cpu().numpy()
        if len(keypoints) > 10:
            # ดึงจุด (ซ้าย)
            p_shoulder = keypoints[5][:2]
            p_elbow    = keypoints[7][:2]
            p_wrist    = keypoints[9][:2]
            p_hip      = keypoints[11][:2]
            p_knee     = keypoints[13][:2]
            
            # วาดจุดสำคัญลงบนภาพ (เฉพาะจุดที่ใช้)
            for point in [p_shoulder, p_elbow, p_wrist, p_hip]:
                 cv2.circle(frame, (int(point[0]), int(point[1])), 8, COLOR_PRIMARY, -1)
                 cv2.circle(frame, (int(point[0]), int(point[1])), 12, COLOR_PRIMARY, 2)

            # Logic คำนวณ (เหมือนเดิม)
            current_active = False
            angle = 0
            
            if exercise == "LATERAL RAISE":
                angle = calculate_angle(p_elbow, p_shoulder, p_hip)
                ideal = 85
                if angle < 30: stage = "down"
                if angle > 75 and stage == "down":
                    stage = "up"
                    counter += 1
                current_active = True
                
            elif exercise == "DEADLIFT":
                angle = calculate_angle(p_shoulder, p_hip, p_knee)
                ideal = 160
                if angle > 160: stage = "up"
                if angle < 120 and stage == "up":
                    stage = "down"
                    counter += 1
                current_active = True

            elif exercise == "CHEST PRESS":
                angle = calculate_angle(p_shoulder, p_elbow, p_wrist)
                ideal = 160
                if angle > 150: stage = "up"
                if angle < 90 and stage == "up":
                    stage = "down"
                    counter += 1
                current_active = True

            elif exercise == "OVERHEAD PRESS":
                angle = calculate_angle(p_shoulder, p_elbow, p_wrist)
                ideal = 170
                if angle < 90: stage = "down"
                if angle > 160 and stage == "down":
                    stage = "up"
                    counter += 1
                current_active = True

            if current_active:
                score = 100 - abs(angle - ideal)
                score = max(0, min(score, 100))
                if counter >= 10: set_count += 1; counter = 0

    except Exception as e:
        pass

    # ================= UI DASHBOARD (HUD STYLE) =================
    
    h, w, _ = frame.shape
    
    # 1. แถบ Sidebar ด้านซ้าย (โปร่งแสง)
    draw_rounded_rect(frame, (0, 0), (300, h), (0, 0, 0), 0, 0, 0)
    
    # 2. หัวข้อ Exercise
    cv2.putText(frame, "EXERCISE MODE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
    cv2.putText(frame, exercise, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
    
    # เส้นคั่น
    cv2.line(frame, (20, 110), (280, 110), (100,100,100), 1)
    
    # 3. REPS COUNTER (ตัวเลขใหญ่ๆ)
    cv2.putText(frame, "REPS", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
    cv2.putText(frame, str(counter), (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 3.5, COLOR_PRIMARY, 5)
    
    # 4. SETS
    cv2.putText(frame, "SETS", (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
    cv2.putText(frame, str(set_count), (180, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_TEXT, 2)

    # 5. FORM SCORE (Progress Bar แบบหลอดพลัง)
    cv2.putText(frame, "FORM ACCURACY", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
    
    # กรอบบาร์
    cv2.rectangle(frame, (20, 320), (280, 350), (50,50,50), -1)
    
    # เนื้อบาร์ (เปลี่ยนสีตามคะแนน)
    bar_width = int((score / 100) * 260)
    bar_color = COLOR_WARNING # สีแดง
    if score > 50: bar_color = COLOR_PRIMARY # สีฟ้า
    if score > 80: bar_color = COLOR_SECONDARY # สีเขียว
    
    cv2.rectangle(frame, (20, 320), (20 + bar_width, 350), bar_color, -1)
    cv2.putText(frame, f"{int(score)}%", (230, 342), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # 6. STAGE INDICATOR (UP/DOWN)
    cv2.putText(frame, "STATUS", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
    
    status_color = (100,100,100)
    status_text = stage.upper()
    if stage == "up": status_color = COLOR_SECONDARY
    elif stage == "down": status_color = COLOR_PRIMARY
    
    # กล่องสถานะ
    cv2.rectangle(frame, (20, 420), (150, 470), status_color, -1)
    cv2.putText(frame, status_text, (35, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

    # 7. ปุ่มแนะนำ
    cv2.putText(frame, "[1-4] Change Mode", (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
    cv2.putText(frame, "[X] Exit Program", (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

    cv2.imshow(window_name, frame)

    # Control
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27: break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
    
    if key == ord('1'): exercise = "LATERAL RAISE"; counter=0
    elif key == ord('2'): exercise = "DEADLIFT"; counter=0
    elif key == ord('3'): exercise = "CHEST PRESS"; counter=0
    elif key == ord('4'): exercise = "OVERHEAD PRESS"; counter=0
    elif key == ord('r'): counter=0; set_count=0

cap.release()
cv2.destroyAllWindows()