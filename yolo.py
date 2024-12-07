from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load your pre-trained YOLOv8 model (replace with the path to your model)
model = YOLO('yolo.py')

# Function to process an image array with YOLO
def process_yolo_image(image_array):
    # Perform inference on the image
    results = model(image_array)

    # Define the class mapping: replace 'kamizelka' (ID 0) with 'vest' and 'kask' (ID 1) with 'helmet'
    class_map = {0: 'vest', 1: 'helmet'}

    # Load the original image for drawing boxes
    image_with_boxes = image_array.copy()

    # Iterate over detected boxes and draw bounding boxes for relevant classes
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Get the class ID of the detected object
            if class_id in class_map:  # Check if the class is in the mapping
                label = class_map[class_id]  # Get the mapped class name ('vest' or 'helmet')
                conf = box.conf[0]  # Confidence score
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

                # Draw the bounding box
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # Add the label and confidence score
                cv2.putText(image_with_boxes, f'{label}: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert BGR to RGB for displaying in Matplotlib
    image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)

    return image_rgb
