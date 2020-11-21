import cv2
import numpy as np
from time import time


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

ret, frame1 = cap.read()
ret, frame2 = cap.read()
times = []
counted = []

for i in range(video_length - 1):
    start = time()

    diff = cv2.absdiff(frame1, frame2)
    # diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
    # kernel = np.ones((5, 5), np.uint8)
    # diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('', diff)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    counter = 0

    for contour in contours:
        # center - x+w/2, y+h/2
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) > 900:
            x_center = int(x + w / 2)
            y_center = int(y + h / 2)
            if w > 50 and h > 100:
                cv2.rectangle(diff, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(diff, (x_center, y_center), 5, (0, 0, 255))
                counter += 1
            # cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
            #             1, (0, 0, 255), 3)
    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    cv2.putText(frame1, str(counter), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3)

    image = cv2.resize(frame1, (1280, 720))
    cv2.imshow("feed", diff)
    frame1 = frame2
    ret, frame2 = cap.read()

    end = time()
    times.append(end - start)
    counted.append(counter)

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()

print(f'Avarage time per frame: {np.mean(times)}s')
print(f'Max people counted in single frame: {np.max(counted)}')
