from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time
import json

app = Flask(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
camera = cv2.VideoCapture(0)
# Инициализация MediaPipe
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Счетчик
counter = 0

def calculate_angle(a, b, c):
    a = np.array(a)  # Первая точка
    b = np.array(b)  # Вторая точка
    c = np.array(c)  # Третья точка

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def process_frame(frame, stage, counter):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        l_angle = calculate_angle(l_hip, l_knee, l_ankle)

        cv2.putText(image, str(l_angle), 
                           tuple(np.multiply(l_knee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

        if l_angle < 75:
            stage = "down"
        if l_angle > 160 and stage == 'down':
            stage = "up"
            counter += 1
            print(counter)

    
        
        

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # Обновление stage
    

    except Exception as e:
        print(e)

    return image, stage ,counter

def generate_frames():
    global counter, stage
    stage = None
    counter = 0
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame, stage, counter = process_frame(frame, stage, counter)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stage_counter')
def stage_counter():
    def event_stream():
        while True:
            yield f"data: {json.dumps({'stage': stage, 'counter': counter})}\n\n"
            time.sleep(1)  # Отправляем обновления каждую секунду
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/connect_camera', methods=['POST'])
def connect_camera():
    global camera
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)
        generate_frames()
    return "Камера подключена"

@app.route('/disconnect_camera', methods=['POST'])
def disconnect_camera():
    global camera
    if camera.isOpened():
        camera.release()
        cv2.destroyAllWindows()
    return "Камера отключена"

if __name__ == "__main__":
    app.run(debug=True)