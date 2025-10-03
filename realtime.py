import cv2
import numpy as np
import tensorflow as tf
import time
import re

# --- Configuration ---
# Path to the TensorFlow SavedModel directory
MODEL_DIR = 'saved_model'
# Path to the label map file
LABEL_MAP_PATH = 'label_map.pbtxt'
# Minimum confidence threshold for detections
CONF_THRESHOLD = 0.4

# --- Function to Load Label Map ---
def load_label_map(label_map_path):
    """
    Parses the user's specific .pbtxt label map file.
    """
    category_index = {}
    try:
        with open(label_map_path, 'r') as f:
            content = f.read()
            
            # Regex tailored for the user's format: name:'...' and id:...
            items = re.findall(r'item\s*{\s*name:\s*\'([^\']+)\'\s*id:\s*(\d+)\s*}', content)

            if not items:
                print(f"Warning: Could not find any valid 'item' entries in '{label_map_path}'.")
                print("-> Please check if the file content format is correct.")
                return {}

            for item in items:
                # The regex captures the name (string) first, then the id (number).
                class_name = item[0]
                class_id = int(item[1])
                category_index[class_id] = class_name
                
        print(f"✅ Successfully loaded {len(category_index)} classes from {label_map_path}")

    except FileNotFoundError:
        print(f"Error: Label map file not found at '{label_map_path}'")
    except Exception as e:
        print(f"An unexpected error occurred while parsing the label map file: {e}")

    return category_index

# --- Load Labels and Prepare Colors ---
category_index = load_label_map(LABEL_MAP_PATH)
if not category_index:
    print("Could not load labels. Exiting.")
    exit()

# Create a color for each possible class ID
max_id = max(category_index.keys()) if category_index else 0
colors = np.random.uniform(0, 255, size=(max_id + 1, 3))


# --- Load the TensorFlow SavedModel ---
print("Loading model from disk...")
try:
    model = tf.saved_model.load(MODEL_DIR)
    infer = model.signatures['serving_default']
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Initialize Video Capture ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("\nStarting video stream... Press 'q' to quit.")
prev_frame_time = 0

# --- Main Loop for Real-Time Detection ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    
    # Pre-process the frame for the model
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(rgb_frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    detections = infer(input_tensor)

    # Extract detection results
    num_detections = int(detections.pop('num_detections'))
    scores = detections['detection_scores'][0].numpy()
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int64)

    # Process each detection
    for i in range(num_detections):
        if scores[i] >= CONF_THRESHOLD:
            class_id = classes[i]
            
            # Denormalize bounding box coordinates
            ymin, xmin, ymax, xmax = boxes[i]
            (left, right, top, bottom) = (xmin * width, xmax * width, ymin * height, ymax * height)
            
            # Look up class name from our dictionary
            label = category_index.get(class_id, f'ID {class_id}')
            display_text = f"{label}: {scores[i]:.2f}"
            
            color = colors[class_id] if class_id < len(colors) else (255, 255, 255)

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), color, 2)
            cv2.putText(frame, display_text, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Calculate and display FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the final frame
    cv2.imshow('Real-time Sign Language Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("Cleaning up and closing...")
cap.release()
cv2.destroyAllWindows()