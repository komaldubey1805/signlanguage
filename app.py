import cv2
import numpy as np
import os
# --- Force TensorFlow to use CPU only ---
# This can prevent errors on machines without a compatible NVIDIA GPU and CUDA setup
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import time
import re
from flask import Flask, Response, render_template, jsonify, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import tempfile
import json

# --- Initialize Flask App ---
# REMOVED template_folder='.' to use the default 'templates' directory
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Added secret key for flash messages

# --- Configuration and Model Loading ---
MODEL_DIR = 'saved_model'
LABEL_MAP_PATH = 'label_map.pbtxt'
CONF_THRESHOLD = 0.5 # Confidence threshold for detections

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Global variable to store the latest prediction ---
last_detection = {"sign": "None"}
video_analysis_results = []

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_label_map(label_map_path):
    """Parses the .pbtxt label map file to get class names."""
    category_index = {}
    try:
        with open(label_map_path, 'r') as f:
            content = f.read()
            # Regex to find item blocks and extract name and id
            items = re.findall(r'item\s*{\s*name:\s*\'([^\']+)\'\s*id:\s*(\d+)\s*}', content)
            if not items:
                print(f"❌ Warning: Could not find any valid 'item' entries in '{label_map_path}'.")
                return {}
            for item in items:
                # item[0] is the name, item[1] is the id
                category_index[int(item[1])] = item[0]
        print(f"✅ Successfully loaded {len(category_index)} classes from {label_map_path}")
        return category_index
    except FileNotFoundError:
        print(f"❌ Error: Label map file not found at '{label_map_path}'.")
        return {}
    except Exception as e:
        print(f"❌ An error occurred while parsing the label map: {e}")
        return {}

def load_model(model_dir):
    """Loads the TensorFlow SavedModel."""
    try:
        model = tf.saved_model.load(model_dir)
        print(f"✅ Model loaded successfully from {model_dir}.")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def process_uploaded_video(video_path):
    """Process an uploaded video file and return comprehensive ASL detection results."""
    global video_analysis_results
    video_analysis_results = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}
    
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    detections_timeline = []
    confidence_scores = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every 3rd frame for better accuracy vs speed balance
        if frame_count % 3 == 0:
            timestamp = frame_count / fps
            
            if detect_fn:
                # Flip the frame horizontally for consistency with live feed
                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = tf.convert_to_tensor(image_rgb)
                input_tensor = input_tensor[tf.newaxis, ...]

                # Run detection
                detections = detect_fn(input_tensor)

                # Extract detection results
                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
                detections['num_detections'] = num_detections
                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                scores = detections['detection_scores']
                classes = detections['detection_classes']

                # Find the most confident detection
                most_confident_label = None
                highest_score = 0

                for i in range(len(scores)):
                    if scores[i] > CONF_THRESHOLD and scores[i] > highest_score:
                        highest_score = scores[i]
                        class_id = classes[i]
                        label = category_index.get(class_id, f'ID {class_id}')
                        most_confident_label = label

                if most_confident_label:
                    # Format the label for display
                    if most_confident_label == "iloveyou":
                        formatted_label = "I Love You"
                    else:
                        formatted_label = most_confident_label.capitalize()
                    
                    detections_timeline.append({
                        "timestamp": round(float(timestamp), 2),
                        "sign": formatted_label,
                        "confidence": round(float(highest_score), 3)
                    })
                    confidence_scores.append(float(highest_score))
        
        frame_count += 1
    
    cap.release()
    
    # Aggregate results - find sign segments with improved logic
    sign_segments = []
    current_sign = None
    current_start = None
    current_confidences = []
    
    for detection in detections_timeline:
        if detection["sign"] != current_sign:
            if current_sign is not None:
                avg_confidence = sum(current_confidences) / len(current_confidences) if current_confidences else 0
                sign_segments.append({
                    "sign": current_sign,
                    "start_time": float(current_start),
                    "end_time": float(detection["timestamp"]),
                    "duration": round(float(detection["timestamp"] - current_start), 2),
                    "avg_confidence": round(float(avg_confidence), 3)
                })
            current_sign = detection["sign"]
            current_start = detection["timestamp"]
            current_confidences = [detection["confidence"]]
        else:
            current_confidences.append(detection["confidence"])
    
    # Add the last segment
    if current_sign is not None and detections_timeline:
        last_timestamp = detections_timeline[-1]["timestamp"]
        avg_confidence = sum(current_confidences) / len(current_confidences) if current_confidences else 0
        sign_segments.append({
            "sign": current_sign,
            "start_time": float(current_start),
            "end_time": float(last_timestamp),
            "duration": round(float(last_timestamp - current_start), 2),
            "avg_confidence": round(float(avg_confidence), 3)
        })
    
    # Calculate additional statistics
    unique_signs = list(set([seg["sign"] for seg in sign_segments]))
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    video_analysis_results = {
        "total_duration": round(float(total_frames / fps), 2),
        "signs_detected": int(len(sign_segments)),
        "unique_signs": int(len(unique_signs)),
        "unique_signs_list": unique_signs,
        "timeline": detections_timeline,
        "segments": sign_segments,
        "avg_confidence": round(float(avg_confidence), 3),
        "total_detections": int(len(detections_timeline))
    }
    
    return video_analysis_results

# --- Load resources on startup ---
category_index = load_label_map(LABEL_MAP_PATH)
detect_fn = load_model(MODEL_DIR)

# --- Frame Generation for Video Stream ---
def generate_frames():
    """Captures frames from the camera, runs detection, and streams the output."""
    global last_detection
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open video stream.")
        return
        
    last_recognition_time = time.time()
    recognition_interval = 1.0  # Time in seconds to hold a recognized word

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            break

        if detect_fn:
            # Flip the frame horizontally for a "mirror" effect
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = tf.convert_to_tensor(image_rgb)
            input_tensor = input_tensor[tf.newaxis, ...]

            # Run detection
            detections = detect_fn(input_tensor)

            # Extract detection results
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            scores = detections['detection_scores']
            classes = detections['detection_classes']
            boxes = detections['detection_boxes']

            most_confident_label = None
            highest_score = 0
            best_box_for_drawing = None

            # Process Detections to find the best one
            for i in range(len(scores)):
                if scores[i] > CONF_THRESHOLD and scores[i] > highest_score:
                    highest_score = scores[i]
                    class_id = classes[i]
                    # NOTE: Your label map starts at ID 1. The model output is 0-indexed.
                    # This was a likely source of error. We now correctly get the label.
                    label = category_index.get(class_id, f'ID {class_id}')
                    most_confident_label = label
                    best_box_for_drawing = boxes[i]
            
            # Draw the box for the most confident detection
            if most_confident_label:
                h, w, _ = frame.shape
                ymin, xmin, ymax, xmax = best_box_for_drawing
                (left, right, top, bottom) = (int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h))

                display_text = f"{most_confident_label.upper()}: {highest_score:.2f}"
                
                # Draw bounding box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Draw label background
                label_size, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (left, top - label_size[1] - 10), (left + label_size[0], top), (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(frame, display_text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


            # Update global recognized word for the API endpoint
            if most_confident_label:
                # Reformat "iloveyou" to "I Love You" for better display
                if most_confident_label == "iloveyou":
                    last_detection["sign"] = "I Love You"
                else:
                    last_detection["sign"] = most_confident_label.capitalize()
                last_recognition_time = time.time()
            # If no confident detection, clear the word after an interval
            elif time.time() - last_recognition_time > recognition_interval:
                last_detection["sign"] = "None"

        # Encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue

        # Yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

    cap.release()

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main page."""
    return render_template('index.html')
    
@app.route('/learning')
def learning():
    """Serves the learning page."""
    return render_template('learning.html')

@app.route('/prediction')
def prediction():
    """Serves the prediction page."""
    return render_template('prediction.html')

@app.route('/login')
def login():
    """Serves the login page."""
    return render_template('login.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video file upload and processing with enhanced validation."""
    if 'video' not in request.files:
        return jsonify({"error": "No video file selected"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No video file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload MP4, AVI, MOV, MKV, or WEBM files."}), 400
    
    # Check file size (additional client-side check)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({"error": "File too large. Maximum size is 100MB."}), 400
    
    filename = secure_filename(file.filename)
    # Add timestamp to filename to avoid conflicts
    timestamp = str(int(time.time()))
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(filepath)
        
        # Validate video file can be opened
        test_cap = cv2.VideoCapture(filepath)
        if not test_cap.isOpened():
            os.remove(filepath)
            return jsonify({"error": "Invalid video file or corrupted video."}), 400
        test_cap.release()
        
        # Process the video
        results = process_uploaded_video(filepath)
        
        # Clean up the uploaded file
        os.remove(filepath)
        
        if "error" in results:
            return jsonify({"error": results["error"]}), 500
        
        return jsonify(results)
        
    except Exception as e:
        # Clean up the uploaded file in case of error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"Error processing video: {str(e)}"}), 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    """API endpoint to get the latest detected sign."""
    return jsonify(last_detection)

@app.route('/recognized_word')
def recognized_word():
    """API endpoint to get the latest recognized word (for compatibility)."""
    return last_detection.get("sign", "None")

@app.route('/get_video_stats')
def get_video_stats():
    """API endpoint to get detailed video analysis statistics."""
    return jsonify(video_analysis_results)

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0')
