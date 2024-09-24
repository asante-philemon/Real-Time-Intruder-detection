import threading
import winsound
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Define paths
dataset_path = r"C:\Users\ASANTE PHILEMON\Desktop\Richard\ImageBasic"
snapshot_path = r"C:\Users\ASANTE PHILEMON\Desktop\Richard\Snapshots" 

# Create directory for snapshots if it doesn't exist
if not os.path.exists(snapshot_path):id

os.makedirs(snapshot_path)

# Load images and their encodings
images = []
classNames = []
myList = os.listdir(dataset_path)

for cl in myList:
    curImg = cv2.imread(os.path.join(dataset_path, cl))
    images.append(curImg)
    name = os.path.splitext(cl)[0]
    classNames.append(name)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img)
        if len(face_locations) == 0:
            continue
        encode = face_recognition.face_encodings(img, face_locations)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Function to play sound
def play_sound():
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)  # Play sound for recognized face

# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), None, 0.75, 0.75)
    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    face_found = False

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            face_found = True
            name = classNames[matchIndex].upper()

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f"CAPTURED: {name}", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            # Save the snapshot
            snapshot_filename = os.path.join(snapshot_path, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(snapshot_filename, frame)

            threading.Thread(target=play_sound).start()  # Play sound for recognized face
        else:
            face_found = True
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, "WRONG PERSON", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    if not face_found:
        cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Cam", frame)

    key_pressed = cv2.waitKey(30)
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
