import cv2
import sys
import time
import matplotlib.pyplot as plt

from keras.preprocessing.image import img_to_array
import imutils
from keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'    ##
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'    ###

# hyper-parameters for bounding boxes shape  ###
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]    ##

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

video_capture = cv2.VideoCapture(0)
fps = video_capture.get(cv2.CAP_PROP_FPS)
time_for_one_frame = 1/(fps)
time_observed = 0

# for graph
start_time = time.time()
timestamps = []
num_faces = []
timestamp = 0

while True:




    arr = []
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

        time_observed += 0
        cv2.putText(frame, "Total time spent on this exhibit: " + str(time_observed),(0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    for (x, y, w, h) in faces:
        area = (w)*(h)
        arr.append([x for (x,y,w,h) in faces if w*h > 7000])
        if area>7000:
            #arr.append(area)
            #print(area)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(frame, str(area), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            time_observed += (len(arr) * time_for_one_frame)

        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.putText(frame, str(area), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)


        cv2.putText(frame, "Total time spent on this exhibit: "  + str(time_observed),(0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)





    num_near_faces = len([x for (x,y,w,h) in faces if w*h > 7000])
    if num_near_faces == 1:
        msg = "1 person looking right now"
    elif num_near_faces == 0:
        msg = "No one is interested in this exhibit right now!"
    else:
        msg = str(num_near_faces) + " people looking right now"
    timestamp = int(time.time()-start_time)
    if not (timestamp in timestamps):
        timestamps.append(timestamp)
        num_faces.append(num_near_faces)

    cv2.putText(frame, msg,(0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    # Display the resulting frame
    cv2.imshow('Video', frame)




    ##########################################################################################################################################
    frame = video_capture.read()[1]
    #reading the frame
    frame = imutils.resize(frame,width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

 
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)

                # draw the label + probability bar on the canvas
               # emoji_face = feelings_faces[np.argmax(preds)]

                
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 2)
                cv2.putText(frameClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)
#    for c in range(0, 3):
#        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
#        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
#        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)


    # cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    #########################################################################################################################################

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Release Capture
        video_capture.release()
        cv2.destroyAllWindows()
        # Plot Graph
        print("Plotting graph...")
        print(timestamps)
        print(num_faces)
        plt.plot(timestamps, num_faces)
        plt.xlabel("Seconds after "+ str(time.gmtime(start_time).tm_hour)+":"+str(time.gmtime(start_time).tm_min))
        plt.ylabel("Number of people interested")
        plt.axis([0,timestamps[-1]+1,0,num_faces[-1]+1])
        plt.show()
        break
