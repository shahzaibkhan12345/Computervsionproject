import cv2
from ultralytics import YOLO

PET_CLASSES=[0,67,41]
model = YOLO("yolov8n.pt")

cap=cv2.VideoCapture(0)

class_names={
    0:"Person",
    67:"Cell phone",
    41:"Cup"
    }

while True:
    ret,frame = cap.read()
    if not ret:
        break
    results = model(frame)
    
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        
        if class_id in PET_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            label = f"{class_names[class_id]}: {confidence:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Pet Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()