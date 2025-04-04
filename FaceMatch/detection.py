import cv2
import math
import argparse
from ultralytics import YOLO
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

def face_match(img, data_path):  # img_path= location of photo, data_path= location of data.pt
    # getting embedding matrix of the given img
    face, prob = mtcnn(img, return_prob=True)  # returns cropped face and probability

    emb = resnet(face.unsqueeze(0)).detach()  # detech is to make required gradient false

    saved_data = torch.load(data_path)  # loading data.pt file
    embedding_list = saved_data[0]  # getting embedding data
    name_list = saved_data[1]  # getting list of names
    dist_list = []  # list of matched distances, minimum distance is used to identify the person

    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)

    idx_min = dist_list.index(min(dist_list))
    return(name_list[idx_min], min(dist_list))

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-6)', '(6-10)', '(10-15)', '(15-23)', '(23-32)', '(32-45)', '(45-60)', '(60-100)']
genderList = ['Male', 'Female']

ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

face_model = YOLO("Face_Plate_weights/best.pt")

# for images
frame = cv2.imread('test/Lionardo/4.jpeg')
padding = 20

resultImg = np.copy(frame)

results = face_model(frame)[0]

faceBoxes = []

for data in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = data
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    faceBoxes.append([x1, y1, x2, y2])
    # to create the bounding box on plates
    resultImg = cv2.rectangle(resultImg, (x1, y1), (x2, y2), (255, 0, 0), 2)

# test
for faceBox in faceBoxes:
    face = frame[max(0, faceBox[1] - padding):
                 min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                 : min(faceBox[2] + padding, frame.shape[1] - 1)]

    frames = np.copy(frame)

    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
    print(f'Gender: {gender}')

    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    print(f'Age: {age[1:-1]} years')

    cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2, cv2.LINE_AA)


    frames = frames[y1: y2, x1: x2]

    result = face_match(frames, 'data.pt')

    print('Face matched with: ', result[0], 'With distance: ', result[1])

    cv2.putText(resultImg, str(result[0]), (faceBox[2] - faceBox[0], faceBox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 0), 2, cv2.LINE_AA)

cv2.imshow("Detecting age and gender", resultImg)

cv2.waitKey(0)

"""

face_model = YOLO("Face_Plate_weights/best.pt")

video = cv2.VideoCapture(0)

padding = 20
while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg = np.copy(frame)

    results = face_model(frame)[0]

    faceBoxes = []

    for data in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = data
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        faceBoxes.append([x1, y1, x2, y2])
        # to create the bounding box on plates
        resultImg = cv2.rectangle(resultImg, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if not faceBoxes:
        print("No face detected")
    # test
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                                                                    :min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Detecting age and gender", resultImg)
"""