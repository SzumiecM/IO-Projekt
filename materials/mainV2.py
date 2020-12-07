import numpy as np
import cv2
from time import time
from imutils.object_detection import non_max_suppression


# TODO  - zmienne wielkości okienek trackerów?
#           - wyciąganie mediany wielkości kwadratów z kilku sekund???
#       - automatyczny dev <---
#       - ZASTOSOWAĆ DRUGI MECHANIZM WYKRYWANIAAA?? <<<---
#           wydaje się być koniecznością, program totalnie wywala się przy example01_mp4 (nie jest konieczny ale
#           fajnie żeby działał jednak zawsze)
#       - dodać licznik osób na obrazie

class Person:

    def __init__(self, xA, yA, xB, yB, tracker):
        self.tracker = tracker
        self.xA = xA
        self.yA = yA
        self.xB = xB
        self.yB = yB
        self.dev = 40  # HARD CODED AS FUCK, TODO FIX

    def set_postions(self, x, y):
        self.xB = xB + xA - x
        self.yB = yB + yA - y
        self.xA = x
        self.yA = y

    def check_if_its_me(self, xA, yA, xB, yB):
        if self.xA - self.dev < xA < self.xA + self.dev:
            if self.yA - self.dev < yA < self.yA + self.dev:
                # TODO ?? po zrobieniu dla wartości B trackery zaczęły się baaaaardzo duplikować
                return True
        return False


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

cap = cv2.VideoCapture('grupaB1.mpg')
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

sizes = []
avg_size = 0

times = []
counted = []
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
    boxes, weights = hog.detectMultiScale(frame, winStride=(16, 16), hitThreshold=0.1)

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    boxes = non_max_suppression(boxes, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in boxes:
        # x = int(xA + (xB - xA) / 2)
        # y = int(yA + (yB - yA) / 2)
        found = False
        sizes.append((xB - xA, yB - yA))
        avg_size = np.mean(sizes)

        for person in people:
            if person.check_if_its_me(xA, yA, xB, yB):
                found = True
                # tracker = cv2.TrackerKCF_create()
                # tracker.init(frame, (xA, yA, xB - xA, yB - yA))
                # person.tracker = tracker
                break

        if not found and np.mean((xB - xA, yB - yA)) <= avg_size:
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, (xA, yA, xB - xA, yB - yA))
            people.append(Person(xA, yA, xB, yB, tracker))

        # cv2.circle(frame, (x, y), 5, (0, 0, 255))
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0))
        counter += 1

    for person in people:
        (success, box) = person.tracker.update(frame)

        if success:
            (x, y, w, h) = [int(v) for v in box]
            person.set_postions(x, y)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

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
