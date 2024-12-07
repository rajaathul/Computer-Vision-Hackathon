from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Use 'yolov8s.pt' or other variants as needed

# Input image path
image_path = 'D:/sk/1.jpg'

# Perform inference on the image
results = model(image_path)

# Get the detections for the person class (class ID 0)
person_detections = results[0].boxes[results[0].boxes.cls == 0]  # Keep only the boxes for class ID 0 (person)

# Load the original image
image_with_boxes = cv2.imread(image_path)

# Draw bounding boxes on the original image
for box in person_detections:  # Iterate through person detections
    x1, y1, x2, y2 = box.xyxy[0]  # Get bounding box coordinates
    conf = box.conf[0]  # Get confidence score
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert to int
    cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle in blue
    cv2.putText(image_with_boxes, f'Person: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Convert BGR to RGB for Matplotlib display
image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)

# Display the image with detections
plt.imshow(image_rgb)
plt.axis('off')  # Hide axes
plt.show()
