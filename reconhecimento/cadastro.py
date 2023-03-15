from ddetector import *
import cv2, pickle
import numpy as np

face_detector   = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# use with compute. 
face_descriptor = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

cam = cv2.VideoCapture(2)
cv2.namedWindow("Screen", cv2.WINDOW_GUI_NORMAL)
face_descs = []
samples =[]
f_descriptions = []
current_id = 0
limit = .5

# trick
def best_fits(fd, fds):
    temp = []
    for sfd in fds:
        diferences = np.linalg.norm(fd - sfd, axis=1)
        minindex = np.argmin(diferences)
        temp.append(sfd[minindex])
    return np.array(temp)


while True:
    try:
        check, frame = cam.read()
        detections = get_detections(frame, face_detector)
        #no detections then we'll be going to do nothing
        if len(detections) == 0:
            continue
        detections = convert_cv2rect_to_dlibrects(detections)
        shape_points = get_shape_points(frame, shape_predictor, detections)
        chip  = dlib.get_face_chip(frame, shape_points[0])

        face_description = face_descriptor.compute_face_descriptor(chip)
        face_description = np.array(face_description)
        
        if len(f_descriptions) == 0:
            f_descriptions.append(np.array([face_description]))
            samples.append(chip)
            current_id += 1
        else:
            if len(f_descriptions) == 1:
                bf = best_fits(face_description, f_descriptions)
                diference = np.linalg.norm(face_description - bf[0])
                if diference < limit:
                    if len(f_descriptions[0]) < 6:
                        f_descriptions[0] = np.concatenate([f_descriptions[0], [face_description]])
                else:
                    print(f"recognized subject={0}")
                    f_descriptions.append(np.array([face_description]))
                    samples.append(chip)
                    current_id += 1
                    
            else:
                bf = best_fits(face_description, f_descriptions)
                diferences = np.linalg.norm(face_description - bf[1:], axis=1)
                minindex = np.argmin(diferences)
                minval = diferences[minindex]

                if minval < limit:
                    print(f"recognized subject={minindex+1}")
                    if len(f_descriptions[0]) < 6:
                        f_descriptions[0] = np.concatenate([f_descriptions[0], [face_description]])
                    f_descriptions[minindex+1] = np.concatenate([f_descriptions[minindex+1], [face_description]])
                else:
                    f_descriptions.append(np.array([face_description]))
                    samples.append(chip)
                    current_id += 1
        cv2.imshow("Screen", chip)
        key = cv2.waitKey(1)
        if key == 27: #ESC
            break
    except KeyError:
        break
 

for i, img in enumerate(samples):
    cv2.imwrite(f"{i}.jpeg",img)

with open("dados.bin","wb+") as data:
    pickle.dump(f_descriptions, data)

print("saved in the file dados.bin")
