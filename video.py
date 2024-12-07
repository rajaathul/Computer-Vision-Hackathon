from ultralytics import YOLO
import cv2

# Load the YOLOv8n model for person detection
person_model = YOLO('yolov8n.pt')
vest_helmet_model = YOLO(r"vest.pt")


class_map = {0: 'vest', 1: 'helmet'}

# Function to check if two boxes overlap
def boxes_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    # Check if the boxes overlap
    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

video_path = r"uploads\a-day-in-the-life-of-construction-worker---how-china-builds_online-video-cutter.com.mp4"  # Specify your video path here
cap = cv2.VideoCapture(video_path)

# Get video details for saving output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define VideoWriter to save the output video
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Process the video frame by frame
frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    print(f"Processing frame {frame_num}...")

    # Perform inference on the frame for person detection with a confidence threshold
    person_results = person_model(frame, conf=0.2)

    # Perform inference on the frame for vest and helmet detection
    vest_helmet_results = vest_helmet_model(frame, conf=0.3)

    # Store person, helmet, and vest bounding boxes
    person_boxes = []
    helmet_boxes = []
    vest_boxes = []

    # Person Detection: Iterate over person results and draw red bounding boxes
    for person_result in person_results:
        for box in person_result.boxes:
            class_id = int(box.cls[0])  # Get the class ID of the detected object
            if class_id == 0:  # Class 0 is 'person' in YOLOv8n
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                person_boxes.append([x1, y1, x2, y2])
                # Draw red bounding box for person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for person
                # Add the label 'Person X'
                cv2.putText(frame, f'Person {len(person_boxes)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Vest and Helmet Detection: Iterate over results and draw blue bounding boxes
    for vest_helmet_result in vest_helmet_results:
        for box in vest_helmet_result.boxes:
            class_id = int(box.cls[0])  # Get the class ID of the detected object
            if class_id in class_map:  # Check if class is 'vest' or 'helmet'
                label = class_map[class_id]  # Get label
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                if label == 'vest':
                    vest_boxes.append([x1, y1, x2, y2])
                elif label == 'helmet':
                    helmet_boxes.append([x1, y1, x2, y2])
                # Draw blue bounding box for vest or helmet
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for vest and helmet
                # Add the label and confidence score
                cv2.putText(frame, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Assign safety status and display per person in terminal
    for idx, p_box in enumerate(person_boxes, start=1):
        is_helmet = any([boxes_overlap(p_box, h_box) for h_box in helmet_boxes])
        is_vest = any([boxes_overlap(p_box, v_box) for v_box in vest_boxes])

        # Determine safety level based on overlaps
        if is_helmet and is_vest:
            status = "Safe"
            color = (0, 255, 0)  # Green for Safe
        elif is_helmet or is_vest:
            status = "Partially Safe"
            color = (0, 255, 255)  # Yellow for Partially Safe
        else:
            status = "Not Safe"
            color = (0, 0, 255)  # Red for Not Safe

        # Print safety status in terminal for each person
        print(f"Person {idx}: {status}")

        # Draw safety status text near the person
        x1, y1, x2, y2 = p_box
        y_label = y2 + 30 if y2 + 30 < frame.shape[0] else y2 - 10  # Adjust label position
        cv2.putText(frame, f'{idx}: {status}', (x1, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

    # Write the frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()
