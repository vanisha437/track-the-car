import numpy as np
import cv2
import math


xCords = []
yCords = []

# Create tracker variable using cv2.legacy.TrackerCSRT_create()
tracker = cv2.legacy.TrackerCSRT_create()

confidenceThreshold = 0.1
NMSThreshold = 0.1

modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3.weights'

labelsPath = 'coco.names'

labels = open(labelsPath).read().strip().split('\n')

yoloNetwork = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

video = cv2.VideoCapture("car.mp4")

detected = False


def drawBox(img, bbox):
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])

    cv2.rectangle(img, (x, y), ((x+w), (y+h)), (0, 0, 255), 3, 1)
    cv2.putText(img, "Tracking", (75, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def carTrack(img, bbox):
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])

    c1 = x + int(w/2)
    c2 = y + int(h/2)

    cv2.circle(img, (c1, c2), 2, (0, 0, 255), 5)

    xCords.append(c1)
    yCords.append(c2)

    for i in range(len(xCords)-1):
        cv2.circle(img, (xCords[i], yCords[i]), 2, (0, 0, 255), 5)


while True:
    check, image = video.read()

    if check:
        image = cv2.resize(image, (0, 0), fx=1, fy=1)
        dimensions = image.shape[:2]
        H, W = dimensions

        if detected == False:
            blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416))
            yoloNetwork.setInput(blob)

            layerName = yoloNetwork.getUnconnectedOutLayersNames()
            layerOutputs = yoloNetwork.forward(layerName)

            boxes = []
            confidences = []
            classIds = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]

                    if confidence > confidenceThreshold:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY,  width, height) = box.astype('int')
                        x = int(centerX - (width/2))
                        y = int(centerY - (height/2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIds.append(classId)

            detectionNMS = cv2.dnn.NMSBoxes(
                boxes, confidences, confidenceThreshold, NMSThreshold)

            if (len(detectionNMS) > 0):
                for i in detectionNMS.flatten():
                    if labels[classIds[i]] == "car":
                        x, y, w, h = boxes[i]
                        color = (255, 0, 0)
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        # Initialize the tracker
                        tracker.init(image, boxes[i])
                        # Set detected variable to true
                    detected = True
        else:
            # Get the trackerInfor from tracker.update(image)
            trackerInfo = tracker.update(image)

            # Store trackerInfor[0] in success and trackInfo[1] in bbox variable
            success = None
            bbox = None

            # If success is true the call the drawBox function with the image and bbox variables
            if success:
                drawBox(image, bbox)
            # else add text "Lost" on the screen and set the detected variable to false
            else:
                cv2.putText(image, "Lost", (75, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                detected = False
            carTrack(image, bbox)

        cv2.imshow('Image', image)
        cv2.waitKey(1)

    key = cv2.waitKey(25)
    if key == 32:
        print("Stopped")
        break
