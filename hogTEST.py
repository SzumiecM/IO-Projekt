import cv2
import numpy as np
from time import time
from imutils.object_detection import non_max_suppression


class Person:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def set_postition(self, x, y):
        self.x = x
        self.y = y


cap = cv2.VideoCapture('grupaB1.mpg')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



times = []
counted = []

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for i in range(video_length - 1):
    start = time()
    ret, image = cap.read()
    #orig = image.copy()
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)
    counter = 0
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.circle(image, (int(x + w / 2), int(y + h / 2)), 5, (0, 0, 255))
        counter += 1

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    # rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    # pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    #
    # # draw the final bounding boxes
    # for (xA, yA, xB, yB) in pick:
    #     cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    #     counter+=1

    end = time()
    times.append(end - start)
    counted.append(counter)

    cv2.imshow("Before NMS", image)
    #cv2.imshow("After NMS", image)

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()

print(f'Avarage time per frame: {np.mean(times)}s')
print(f'Max people counted in single frame: {np.max(counted)}')
