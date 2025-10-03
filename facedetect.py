
import cv2
import os
import time

"""Use harrcascades"""""
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

"""use webcam"""
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Cannot open webcam")
    exit()

    """screenshot """

def save_screenshot(frame, output_dir="screenshots"):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"screenshot_{int(time.time())}.jpg")
    cv2.imwrite(filename, frame)
    print(f" Screenshot saved: {filename}")


cv2.namedWindow("Face Detection - Press 'q' to quit", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Detection - Press 'q' to quit", 1280, 720) 

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) > 0:
        save_screenshot(frame)

    """use rectangles after face detection"""
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    """ face count """
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Face Detection - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  





























#         run_with_matplotlib()
