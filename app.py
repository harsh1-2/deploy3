
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from flask import Flask, render_template, Response, request, redirect
import time
import os  # ✅ ADDED

app = Flask(__name__)

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize points
wpoints = [deque(maxlen=1024)]  # White color for drawing

# White color definition
red_color = (0, 0, 255)
paintWindow = np.ones((471, 636, 3)) * 255  # White canvas

# MediaPipe hands initialization
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Drawing state flags
drawing_active = False  # Flag for when "OK" sign is detected

# Buffer for hand detection consistency
hand_detection_buffer = 0
last_hand_count = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index2')
def index2():
    return render_template('index2.html')

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        return redirect('/index2')  # Redirect to video feed after login
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/paintboard_feed')
def paintboard_feed():
    ret, buffer = cv2.imencode('.jpg', paintWindow)
    if not ret:
        return '', 204  # No content if encoding fails
    paintboard = buffer.tobytes()
    return Response(paintboard, mimetype='image/jpeg')

@app.route('/toggle_eraser', methods=['POST'])
def toggle_eraser():
    global is_eraser_active
    is_eraser_active = not is_eraser_active  # Toggle eraser state
    return '', 204

@app.route('/update_finger_position', methods=['POST'])
def update_finger_position():
    data = request.get_json()
    finger_x = data.get('x')
    finger_y = data.get('y')
    return '', 204

def gen_frames():
    global paintWindow
    global drawing_active
    global wpoints

    global hand_detection_buffer
    global last_hand_count

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)

        # MediaPipe hand detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        finger_x, finger_y = None, None
        hand_count = 0  # Count how many hands are detected

        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                finger_x, finger_y = int(index_tip.x * w), int(index_tip.y * h)

                # Check for "OK" sign
                distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
                if distance < 0.05:
                    drawing_active = True
                else:
                    drawing_active = False

                if drawing_active:
                    wpoints.append(deque(maxlen=1024))
                    wpoints[-1].appendleft((finger_x, finger_y))
                else:
                    wpoints[-1].appendleft((finger_x, finger_y))

        # Handle erasing if two hands are detected consistently
        if hand_count == 2:
            if last_hand_count == 2:
                hand_detection_buffer += 1
            else:
                hand_detection_buffer = 0
        else:
            hand_detection_buffer = 0

        if hand_detection_buffer > 10:
            wpoints = [deque(maxlen=1024)]
            paintWindow = np.ones((471, 636, 3)) * 255

        last_hand_count = hand_count

        for j in range(len(wpoints)):
            for k in range(1, len(wpoints[j])):
                if wpoints[j][k - 1] is None or wpoints[j][k] is None:
                    continue
                cv2.line(frame, wpoints[j][k - 1], wpoints[j][k], red_color, 2)
                cv2.line(paintWindow, wpoints[j][k - 1], wpoints[j][k], red_color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # ✅ Corrected key name to "PORT"
    app.run(host='0.0.0.0', port=port)
