import cv2
from ultralytics import YOLO
import os

# Load the YOLO models (person and fine-tuned vest/helmet model)
person_model = YOLO('yolov8n.pt')
vest_helmet_model = YOLO(r"D:/StudyCU/Tri5/Computer Vision/sk/best (1).pt")  # Update the path to your fine-tuned model

def detect_and_process(image_path):
    # Perform person detection using the YOLO model
    person_results = person_model(image_path, conf=0.2)
    
    # Perform vest and helmet detection using the fine-tuned YOLO model
    vest_helmet_results = vest_helmet_model(image_path, conf=0.5)

    # Load the image to draw bounding boxes
    image_with_boxes = cv2.imread(image_path)

    # Store bounding boxes
    person_boxes = []
    helmet_boxes = []
    vest_boxes = []

    # Detect persons
    for person_result in person_results:
        for box in person_result.boxes:
            class_id = int(box.cls[0])
            if class_id == 0:  # YOLO class 0 corresponds to person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append([x1, y1, x2, y2])
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for person
                cv2.putText(image_with_boxes, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Detect vests and helmets
    class_map = {0: 'vest', 1: 'helmet'}
    for vest_helmet_result in vest_helmet_results:
        for box in vest_helmet_result.boxes:
            class_id = int(box.cls[0])
            label = class_map[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if label == 'vest':
                vest_boxes.append([x1, y1, x2, y2])
            elif label == 'helmet':
                helmet_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for vest/helmet
            cv2.putText(image_with_boxes, label.capitalize(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Define a function to check if bounding boxes overlap
    def boxes_overlap(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

    # Initialize counters for helmet and vest usage
    num_safe = 0
    num_partially_safe = 0
    num_not_safe = 0

    # Check overlaps and determine safety
    for p_box in person_boxes:
        is_helmet = any([boxes_overlap(p_box, h_box) for h_box in helmet_boxes])
        is_vest = any([boxes_overlap(p_box, v_box) for v_box in vest_boxes])

        # Determine status based on overlaps
        if is_helmet and is_vest:
            num_safe += 1
        elif is_helmet or is_vest:
            num_partially_safe += 1
        else:
            num_not_safe += 1

    # Save the processed image in the uploads folder
    output_path = os.path.join('uploads', 'output_image_with_detections.jpg')
    cv2.imwrite(output_path, image_with_boxes)

    # Return the path to the saved image and safety counts
    total_people = len(person_boxes)
    return output_path, {
        "output_image_path": output_path,
        "total_people": total_people,
        "safe_count": num_safe,
        "partially_safe_count": num_partially_safe,
        "not_safe_count": num_not_safe
    }
