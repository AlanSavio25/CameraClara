import cv2
import sys
import time


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

video_capture = cv2.VideoCapture(0)
arr = []
while True:
    # Capture frame-by-frame

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags= cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    if len(faces) == 0:
        msg = "No one is interested in this exhibit right now!"
        cv2.putText(frame, msg,(0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    for (x, y, w, h) in faces:
        area = (w)*(h)
        if area>10000:
            arr.append(area)
            print(area)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(frame, str(area), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.putText(frame, str(area), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)

        num_near_faces = len([x for (x,y,w,h) in faces if w*h > 10000])


        if num_near_faces == 1:
            msg = "1 person looking right now"
        elif num_near_faces == 0:
            msg = "No one is interested in this exhibit right now!"
        else:
            msg = str(num_near_faces) + " people looking right now"
        cv2.putText(frame, msg,(0,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
