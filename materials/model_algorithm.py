import cv2
from time import time
import numpy as np

thres = 0.45  # Threshold to detect object

cap = cv2.VideoCapture('grupaB1.mpg')
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

times = []
counted = []

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for i in range(video_length - 1):
    start = time()
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    # print(classIds, bbox)
    counter = 0

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classNames[classId - 1].upper() == 'PERSON':
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                counter += 1
                # cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(img, str(counter), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3)

    cv2.imshow("Output", img)
    cv2.waitKey(1)

    end = time()
    times.append(end - start)
    counted.append(counter)

print(f'Avarage time per frame: {np.mean(times)}s')
print(f'Max people counted in single frame: {np.max(counted)}')