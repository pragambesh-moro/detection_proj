from ultralytics import YOLO
import cv2

#intitalising the model
model = YOLO('yolov8n.pt')

#initialising video capture device
cam_cap = cv2.VideoCapture(0)

#checking if the camera is working or not
if not cam_cap.isOpened():
    print("Error! Could not open camera!! Terminating the process")
    exit()

print("Initialising real-time object detection, press 'q' to quit")

#Main program
while True:
    ret, frame = cam_cap.read()
    if not ret:
        print("Error! Could not read frame. Terminating the process")
        break
    
    #Object detection part, stream=True enables real-time functionality
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) #getting the coordinates
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            #Drawing boxes
            if confidence > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) #getting coordinates, and setting colour to green
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) #adding text

    cv2.imshow('Real-time Object Detection', frame) #displaying the frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Quitting
cam_cap.release()
cv2.destroyAllWindows()
print("Object Detection Stopped")

