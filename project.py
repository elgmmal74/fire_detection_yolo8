import cv2
from ultralytics import YOLO
from playsound import playsound

# Load the model
model = YOLO('best.pt') 

# Specify the path to the video file
video_path = 'video.mp4' 

# Specify the path to the alarm sound file
alarm_sound_path = 'fire_alarm.mp3' 

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Define the class names for filtering
fire_smoke_human_classes = ['fire', 'smoke', 'person']

# Start processing frames
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("End of video file or error reading frame.")
        break
    
    # Perform object detection
    results = model(frame)
    
    # Initialize fire detection flag
    fire_detected = False
    
    # Iterate through the detections
    for result in results:
        for box in result.boxes:
            class_id = box.cls.item()
            confidence = box.conf.item()
            
            # Check if the detected class is fire
            if result.names[class_id] == 'fire':
                fire_detected = True
                
                # Draw the bounding box and label on the frame
                x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
                label = f"{result.names[class_id]}: {confidence:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Red for fire
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # If fire was detected, play the alarm sound
    if fire_detected:
        playsound(alarm_sound_path)
    
    # Display the frame with detections
    cv2.imshow('YOLOv8 Detection', frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close the window
cap.release()
cv2.destroyAllWindows()
