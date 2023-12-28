from flask import Flask, request, jsonify
import cv2
import numpy as np
import json
from tensorflow.lite.python.interpreter import Interpreter

app = Flask(__name__)

# Load TensorFlow Lite model


# Load label map into memory
with open('labelmap.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Function to perform object detection on a frame
def perform_detection(frame):
    interpreter = Interpreter(model_path='detect.tflite')
    interpreter.allocate_tensors()
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Resize the frame to the expected input shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e., if the model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the frame as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence of detected objects

    detected_objects = []
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Adjust the minimum confidence threshold as needed
            ymin, xmin, ymax, xmax = boxes[i]
            ymin = int(max(1, ymin * frame.shape[0]))
            xmin = int(max(1, xmin * frame.shape[1]))
            ymax = int(min(frame.shape[0], ymax * frame.shape[0]))
            xmax = int(min(frame.shape[1], xmax * frame.shape[1]))

            object_name = labels[int(classes[i])]
            confidence = float(scores[i])  # Convert NumPy float32 to Python float
            
            detected_objects.append({
                'object_name': object_name,
                'confidence': confidence,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })

    interpreter.reset_all_variables()
    return detected_objects

# API endpoint for object detection
@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame found in request'})

    file = request.files['frame']
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Failed to decode frame'})

    detected_objects = perform_detection(frame)

    return json.dumps({'detected_objects': detected_objects})
@app.route('/hello')
def hello():
    return 'Hello, World!'
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001,debug=True)
