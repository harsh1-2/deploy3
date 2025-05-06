

#        FINAL CODE


import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from flask import Flask, render_template, Response, request, redirect
import time

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

# @app.route('/select_color/<int:color_index>', methods=['POST'])
# def select_color(color_index):
#     global colorIndex
#     colorIndex = color_index  # Update selected color
#     return '', 204

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
    global wpoints  # Make sure wpoints is defined globally

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
            hand_count = len(results.multi_hand_landmarks)  # Update hand count

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                thumb_tip = hand_landmarks.landmark[4]  
                index_tip = hand_landmarks.landmark[8]  
                h, w, _ = frame.shape
                finger_x, finger_y = int(index_tip.x * w), int(index_tip.y * h)

                # Check if "OK" sign is made (thumb tip close to index tip)
                distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
                if distance < 0.05:  
                    drawing_active = True  # Draw when "OK" sign detected
                else:
                    drawing_active = False  # Stop drawing when "OK" sign not detected

                # Handle drawing
                if drawing_active:
                    wpoints.append(deque(maxlen=1024))  # Start a new line
                    wpoints[-1].appendleft((finger_x, finger_y))
                else:
                    wpoints[-1].appendleft((finger_x, finger_y))

        # Check if two hands are consistently detected
        if hand_count == 2:
            if last_hand_count == 2:
                hand_detection_buffer += 1  # Increment buffer if two hands detected consistently
            else:
                hand_detection_buffer = 0  # Reset buffer if the count changes
        else:
            hand_detection_buffer = 0  # Reset buffer if hand count is less than 2

        # Only clear when two hands are detected consistently for a short period
        if hand_detection_buffer > 10:  # 10 frames (about 1/2 second at ~20 FPS)
            wpoints = [deque(maxlen=1024)]  # Reset the drawing points
            paintWindow = np.ones((471, 636, 3)) * 255  # Clear the paintboard (reset canvas)

        last_hand_count = hand_count  # Store the last hand count for comparison

        # Draw on the paintboard
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
    app.run(debug=True)


