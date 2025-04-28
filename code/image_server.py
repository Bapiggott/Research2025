from flask import Flask, request, jsonify, send_from_directory, render_template_string, redirect
import cv2
import numpy as np
import torch
import os
import json
import time
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from collections import deque
import threading

# Your "second code" import
from mine_test2 import SecondCodeHandler

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
DEPTH_OUTPUT_FOLDER = 'depth_outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DEPTH_OUTPUT_FOLDER'] = DEPTH_OUTPUT_FOLDER

N = 5
latest_detections = deque(maxlen=N)
detections_lock = threading.Lock()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)

# Depth model initialization
depth_model_name = "depth-anything/Depth-Anything-V2-Small-hf"
depth_image_processor = AutoImageProcessor.from_pretrained(depth_model_name)
depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_name).to(device)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEPTH_OUTPUT_FOLDER, exist_ok=True)

# Our "second code" handler instance
second_code_handler = SecondCodeHandler()

# Global list for conversation
conversation = []

# Global dictionary to store the latest processing times
last_times = {
    'yolo_time': 0,
    'depth_time': 0,
    'total_time': 0
}

###############################################################################
# HTML template with chat + dynamic times
###############################################################################
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>YOLO and Depth Estimation Server</title>
    <script>
        function refreshImage() {
            var timestamp = new Date().getTime();
            document.getElementById('latest-image').src = '/image/latest?' + timestamp;
            document.getElementById('depth-image').src = '/depth_image/latest?' + timestamp;
        }
        setInterval(refreshImage, 5000);
    </script>
</head>
<body>
    <h1>YOLO and Depth Estimation Server</h1>

    <!-- Prompt form -->
    <h2>Submit a Prompt</h2>
    <form action="/execute_prompt" method="post">
        <label for="prompt">Enter Prompt:</label><br>
        <input type="text" id="prompt" name="prompt" style="width:300px;" required>
        <input type="submit" value="Run Prompt Code">
    </form>

    <!-- Chat/Conversation History -->
    <h2>Conversation</h2>
    <div id="chat-history" style="border:1px solid #ccc; padding:10px; margin:10px 0; max-width:600px;">
        {% for msg in conversation %}
            <p><strong>{{ msg.role }}:</strong> {{ msg.text }}</p>
        {% endfor %}
    </div>

    <hr>

    <h2>Latest Processed Images:</h2>
    <div style="display: flex;">
        <div>
            <h3>Original Image</h3>
            <img id="latest-image" src="/image/latest" alt="Latest Image" style="max-width: 100%;">
        </div>
        <div>
            <h3>Depth Image</h3>
            <img id="depth-image" src="/depth_image/latest" alt="Depth Image" style="max-width: 100%;">
        </div>
    </div>

    <h3>Processing Times (ms):</h3>
    <p>YOLO Detection: {{ yolo_time }} ms</p>
    <p>Depth Estimation: {{ depth_time }} ms</p>
    <p>Total Time: {{ total_time }} ms</p>
</body>
</html>
'''

@app.route('/')
def home():
    """
    Render the main page (chat, images, times).
    Uses 'last_times' to display the most recent YOLO/depth times.
    """
    return render_template_string(
        HTML_TEMPLATE,
        conversation=conversation,
        yolo_time=last_times['yolo_time'],
        depth_time=last_times['depth_time'],
        total_time=last_times['total_time']
    )

@app.route('/execute_prompt', methods=['POST'])
def execute_prompt():
    """
    1) Get user prompt
    2) Call second_code
    3) Append conversation
    4) Redirect to homepage
    """
    user_prompt = request.form.get('prompt', '').strip()
    if not user_prompt:
        return redirect('/')

    # Call second code
    result = run_second_code(user_prompt)

    # Add to conversation
    conversation.append({'role': 'user', 'text': user_prompt})
    conversation.append({'role': 'assistant', 'text': result})

    # Redirect back so we re-render the template with updated convo
    return redirect('/')

def run_second_code(prompt: str) -> str:
    """
    Use SecondCodeHandler from 'mine_test2.py'
    """
    print(f"[DEBUG] run_second_code() with prompt: {prompt}")
    result = second_code_handler.run(prompt)
    return result

@app.route('/detect', methods=['POST'])
def run_yolo():
    """
    1) Handle image upload
    2) YOLO detection
    3) Depth estimation
    4) Update times
    5) Return updated page
    """
    global last_times
    try:
        if 'image' not in request.files:
            return 'No file part in request', 400

        file = request.files['image']
        if file.filename == '':
            return 'No selected file', 400

        """print("=== /detect: Receiving new image ===")
        print(f"Filename: {file.filename}")"""

        # Decode the image
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # YOLO detection timing
        start_time = time.time()
        results = model_yolo(img)
        yolo_time = (time.time() - start_time) * 1000

        # Convert YOLO results
        detections = results.pandas().xyxy[0].to_dict(orient="records")
        #print(f"YOLO detection found {len(detections)} objects")

        # Save uploaded image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        image_filename = f"{timestamp}_{file.filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        with open(image_path, 'wb') as f:
            f.write(img_bytes)

        # Store detection info
        detection_data = {"timestamp": timestamp, "detections": detections}
        with detections_lock:
            latest_detections.append(detection_data)

        # Depth estimation timing
        depth_time = run_depth_estimation(image_path, image_filename)
        total_time = (time.time() - start_time) * 1000

        # Update global times
        last_times['yolo_time'] = int(yolo_time)
        last_times['depth_time'] = int(depth_time)
        last_times['total_time'] = int(total_time)

        #print(f"YOLO time: {yolo_time:.1f} ms, Depth time: {depth_time:.1f} ms, Total: {total_time:.1f} ms")

        # Return updated page with new times + conversation
        return render_template_string(
            HTML_TEMPLATE,
            conversation=conversation,
            yolo_time=last_times['yolo_time'],
            depth_time=last_times['depth_time'],
            total_time=last_times['total_time']
        )

    except Exception as e:
        # print("[ERROR] Exception in /detect:", e)
        return str(e), 400

def run_depth_estimation(image_path, original_filename):
    """
    Runs the depth model and returns how long it took (ms).
    """
    try:
        start_time = time.time()

        # Load the saved image
        image = Image.open(image_path)
        inputs = depth_image_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Resize depth map
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).cpu()

        depth_array = prediction.squeeze().numpy()

        # Create array [x, y, depth]
        height, width = depth_array.shape
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        depth_with_coordinates = np.stack([x_coords.ravel(), y_coords.ravel(), depth_array.ravel()], axis=1)

        # Save to text + npy
        np.savetxt("depth_with_coordinates.txt", depth_with_coordinates, fmt='%.4f')
        np.save("depth_with_coordinates.npy", depth_with_coordinates)

        # Save depth visualization
        formatted = (depth_array * 255 / np.max(depth_array)).astype("uint8")
        depth_image = Image.fromarray(formatted)
        depth_output_filename = f"depth_{original_filename}.png"
        depth_output_path = os.path.join(app.config['DEPTH_OUTPUT_FOLDER'], depth_output_filename)
        depth_image.save(depth_output_path)

        duration = (time.time() - start_time) * 1000
        return duration

    except Exception as e:
        #print("[ERROR] Depth estimation failed:", e)
        return 0

@app.route('/detections/latest', methods=['GET'])
def get_latest_detection():
    """
    Returns the JSON of the most recent detection data
    """
    with detections_lock:
        if latest_detections:
            return jsonify(latest_detections[-1])
        return jsonify([]), 404

@app.route('/image/latest')
def latest_image():
    """
    Serves the most recently uploaded image
    """
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    if files:
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(app.config['UPLOAD_FOLDER'], x)))
        return send_from_directory(app.config['UPLOAD_FOLDER'], latest_file)
    return '', 404

@app.route('/depth_image/latest')
def latest_depth_image():
    """
    Serves the most recently generated depth image
    """
    files = os.listdir(app.config['DEPTH_OUTPUT_FOLDER'])
    if files:
        latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(app.config['DEPTH_OUTPUT_FOLDER'], x)))
        return send_from_directory(app.config['DEPTH_OUTPUT_FOLDER'], latest_file)
    return '', 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
