import numpy as np
import cv2
from time import time
from imutils.object_detection import non_max_suppression


# TODO  - usuwanie osób które wyszły poza ekran ( tj. zapisywanie ich ale usuwanie z frame'a )
#       - dynamiczne wyliczanie odchylenia ( np. z analizy wymiarów kwadratu (?) )
#           - super byłoby zrobić dynamiczne odchylenie dla każdego z osobna, które wyliczałoby się co jakiś czas

class Person:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def check_if_its_me(self, x, y, devX, devY):
        if self.x - devX < x < self.x + devX:
            if self.y - devY < y < self.y + devY:
                return True
        return False


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

cap = cv2.VideoCapture('example_01.mp4')
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

times = []
counted = []

# TODO do zrobienia plik konfiguracyjny jeśli wyjdzie więcej takich zmiennych
devX = 30
devY = 60
people = []

for i in range(video_length - 1):
    counter = 0
    start = time()

    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    boxes = non_max_suppression(boxes, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in boxes:
        x = int(xA + (xB - xA) / 2)
        y = int(yA + (yB - yA) / 2)
        found = False

        for person in people:
            if person.check_if_its_me(x, y, devX, devY):
                person.set_position(x, y)
                found = True
                break

        if not found:
            people.append(Person(x, y))

        cv2.circle(frame, (x, y), 5, (0, 0, 255))
        counter += 1

    for person in people:
        cv2.circle(frame, (person.x, person.y), 5, (255, 255, 0))

    cv2.imshow('frame', frame)

    end = time()
    times.append(end - start)
    counted.append(counter)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f'Avarage time per frame: {np.mean(times)}s')
print(f'Max people counted in single frame: {np.max(counted)}')
print(f'People counted: {len(people)}')
