
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


























# import cv2
# import os
# import time


# """" haarcascade"""
# def load_classifier():
#     return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# """screenshot """
# def save_screenshot(frame, output_dir="screenshots"):
#     os.makedirs(output_dir, exist_ok=True)
#     filename = os.path.join(output_dir, f"screenshot_{int(time.time())}.jpg")
#     cv2.imwrite(filename, frame)
#     print(f" Screenshot saved: {filename}")

# """" face detect """
# def detect_and_annotate(frame, face_cascade):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30)
#     )

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, "Face Detected", (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        

#         """Show face counts """""


#     cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#     return frame, faces


# def main():
#     face_cascade = load_classifier()
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print(" Cannot open webcam")
#         return
    
#         print("webcam opened successfully ") 

#     cv2.namedWindow("Face Detection - Press 'q' to quit", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Face Detection - Press 'q' to quit", 1280, 720)

#     last_screenshot_time = 0
#     screenshot_interval = 5 

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

        
#         frame = cv2.resize(frame, (1280, 720))

#         annotated_frame, faces = detect_and_annotate(frame, face_cascade)


#         """Screenshot"""

#         if len(faces) > 0 and (time.time() - last_screenshot_time) > screenshot_interval:
#             save_screenshot(annotated_frame)
#             last_screenshot_time = time.time()

#         cv2.imshow("Face Detection", annotated_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print(" program executes...")
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "_main_":
#     main() 





# import cv2
# import os
# import time
# import matplotlib.pyplot  as plt

# # Force Qt backend for OpenCV GUI (helps on Windows)
# os.environ["QT_QPA_PLATFORM"] = "windows"

# # Load Haarcascade
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # Save screenshots with timestamp
# def save_screenshot(frame, output_dir="screenshots"):
#     os.makedirs(output_dir, exist_ok=True)
#     filename = os.path.join(output_dir, f"screenshot_{int(time.time())}.jpg")
#     cv2.imwrite(filename, frame)
#     print(f"📸 Screenshot saved: {filename}")

# # Detect faces and annotate
# def detect_and_annotate(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, "Face", (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#     return frame, faces

# # Main method using OpenCV window
# def run_with_imshow():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("❌ Cannot open webcam")
#         return False

#     print("✅ Webcam opened successfully (OpenCV mode)")
#     last_screenshot_time = 0
#     screenshot_interval = 5  # seconds

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("❌ Failed to grab frame")
#             break

#         frame = cv2.resize(frame, (800, 600))
#         annotated_frame, faces = detect_and_annotate(frame)

#         # Save screenshot if faces detected (interval-based)
#         if len(faces) > 0 and (time.time() - last_screenshot_time) > screenshot_interval:
#             save_screenshot(annotated_frame)
#             last_screenshot_time = time.time()

#         cv2.imshow("Face Detection (Press q to quit)", annotated_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("👋 Exiting...")
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return True

# # Fallback mode with matplotlib (snapshot only)
# def run_with_matplotlib():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Cannot open webcam (matplotlib mode)")
#         return

#     print(" Webcam opened successfully (Matplotlib mode)")
#     ret, frame = cap.read()
#     cap.release()

#     if ret:
#         annotated_frame, faces = detect_and_annotate(frame)

#         # Save screenshot if faces detected
#         if len(faces) > 0:
#             save_screenshot(annotated_frame)

#         frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#         plt.imshow(frame_rgb)
#         plt.title(f"Faces Detected: {len(faces)}")
#         plt.axis("off")
#         plt.show()
#     else:
#         print("❌ Failed to capture frame")

# # Run program
# if __name__ == "_main_":
#     worked = run_with_imshow()
#     if not worked:
#         print("⚠ Falling back to Matplotlib snapshot mode...")
#         run_with_matplotlib()