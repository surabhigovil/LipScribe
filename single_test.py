import cv2
import numpy as np
from PIL import Image
from tensorflow import keras 
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import json 
import os
import sys 

video_path = <video_path_here>

if os.path.exists(video_path):
    print(video_path)
else:
    sys.exit("video path does not exist")

json_file = open('./lip_reading_models/model_D.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./lip_reading_models/model_D.h5")

# set the sizes of mouth region (ROI) -> input shape
WIDTH = 24
HEIGHT = 32
DEPTH = 28

# Haar cascade classifiers - frontal face, profile face and mouth detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')

def video_to_npy_array(video):

    cap = cv2.VideoCapture(video)

    count = 0

    # initialize MedianFlow tracker for tracking mouth region
    medianflow_tracker = cv2.TrackerMedianFlow_create()

    lip_frames = []
    frames = []
    found_first_frame = False
    found_lips = False
    found_face = False
    video_array = None

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            if len(lip_frames) > 0:
                video_array = np.array(lip_frames, dtype="uint8")
            break
        # convert frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not found_lips:
            # use Haar classifier to find the frontal face


            faces = face_cascade.detectMultiScale(gray, 1.2, 3, minSize=(100, 100))
            if len(faces) == 0:
           
                    
                # if frontal face is not found then try to detect profile face
                faces = face_profile_cascade.detectMultiScale(gray, 1.2, 3, minSize=(100, 100))
                if len(faces) == 0:
                    found_face = False
                    if not found_lips:
                    
                        break
                else:
                    found_face = True
            else:
                found_face = True

            if found_face:
                face = faces[0]
                face[3] += 20
                
                for (x,y,w,h) in faces:
                # drawing rectangle for face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                lower_face = int(h * 0.5)
                lower_face_roi = gray[y + lower_face:y + h, x:x + w]

                # detect mouth region in lower half of the face
                mouths = mouth_cascade.detectMultiScale(lower_face_roi, 1.3, 15)
                if len(mouths) > 0:
                    # if first mouth is found
                    mouth = mouths[0]
                    mouth[0] += x  # add face x to fix absolute pos
                    mouth[1] += y + lower_face  # add face y to fix absolute pos

                    m = mouth
                    # drawing rectangle for mouth
                    cv2.rectangle(frame, (m[0], m[1]), (m[0] + m[2], m[1] + m[3]), (0, 255, 0), 2)

                    # initialized the init tracker
                    if not found_lips:
                        lip_track = mouth
                        # extend tracking area
                        lip_track[0] -= 10
                        lip_track[1] -= 20
                        lip_track[2] += 20
                        lip_track[3] += 30
                        medianflow_tracker.init(frame, tuple(lip_track))
                        found_lips = True

                    if count == 0:
                        found_first_frame = True

                    if not found_first_frame:
                        cap = cv2.VideoCapture(video)
                        found_first_frame = True
                        continue

                # skip the sample, if the face is not found
                else:
                    if not found_lips:
                     
                        break
        # Update medianflow tracker
        else:
            ok, bbox = medianflow_tracker.update(frame)
            # if tracker is successfully matched in following frame
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)

                lips_roi = gray[
                           int(bbox[1]):int(bbox[1]) + int(bbox[3]),
                           int(bbox[0]):int(bbox[0]) + int(bbox[2])
                           ]

                # prevent crash when tracker goes out of frame
                # and skip video if this occurs (eg. waved hand in front of mouth...)
                if lips_roi.size == 0:
                    break

                lips_resized = cv2.resize(lips_roi, (HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)
                lip_frames.append(lips_resized)
         

            # if tracker is lost, skip the sample
            else:
             
                break
        
        if len(frames) != DEPTH:
            #cv2.imwrite('outputs/haar/frame-' + str(count) + ".png", frame)
            frames.append(frame)
        count += 1
        if count > DEPTH:
            video_array = np.array(lip_frames, dtype="uint8")
            break

    cap.release()
    return video_array


# while True:
#         _, frame = video.read()

#         #Convert the captured frame into RGB
#         im = Image.fromarray(frame, 'RGB')

#         #Resizing into dimensions you used while training
#         im = im.resize((24,32))
#         img_array = np.array(im)

#         #Expand dimensions to match the 4D Tensor shape.
img_array = video_to_npy_array(video_path)
img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict function using keras
prediction = loaded_model.predict(img_array)#[0][0]
print(prediction)
        # #Customize this part to your liking...
        # if(prediction == 1 or prediction == 0):
        #     print("No Human")
        # elif(prediction < 0.5 and prediction != 0):
        #     print("Female")
        # elif(prediction > 0.5 and prediction != 1):
        #     print("Male")

        # cv2.imshow("Prediction", frame)
        # key=cv2.waitKey(1)
        # if key == ord('q'):
        #         break

# video.release()
# cv2.destroyAllWindows()
