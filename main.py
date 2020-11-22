import numpy as np
import cv2
from time import time
from imutils.object_detection import non_max_suppression


# TODO  - usuwanie osób które wyszły poza ekran ( tj. zapisywanie ich ale usuwanie z frame'a )
#       - dynamiczne wyliczanie odchylenia ( np. z analizy wymiarów kwadratu (?) )
#           - super byłoby zrobić dynamiczne odchylenie dla każdego z osobna, które wyliczałoby się co jakiś czas
#           - porównywanie koloru
#           - CZY WARTO ZASTOSOWAĆ STATYSTYKĘ ( NP. ODCHYLENIA STANDARDOWE ITP ) ????????????

class Person:

    def __init__(self, x, y, dev_x, dev_y, color):
        self.x = x
        self.y = y
        self.dev_x = dev_x
        self.dev_y = dev_y
        self.color = color

        self.updated = False
        self.vel_x = 0
        self.vel_y = 0
        self.dev_color = 100

    def set_position(self, x, y):
        self.vel_x = x - self.x  # rozdzielić na ilość ramek
        self.vel_y = y - self.y
        self.x = x
        self.y = y
        self.updated = True

    def check_if_its_me(self, x, y, color):
        if self.x - self.dev_x < x < self.x + self.dev_x:
            if self.y - self.dev_y < y < self.y + self.dev_y:
                # if self.compare_color(color):
                #     return True
                return True
        return False

    def set_dev(self, dev_x, dev_y):
        self.dev_x = dev_x
        self.dev_y = dev_y

    def update_position(self):
        self.x += self.vel_x
        self.y += self.vel_y

        # TODO da się krócej / lepiej ????
        if self.vel_x > 0:
            self.vel_x -= 1
        elif self.vel_x < 0:
            self.vel_x += 1
        if self.vel_y > 0:
            self.vel_y -= 1
        elif self.vel_y < 0:
            self.vel_y += 1

    #def compare_color(self, color):



# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

cap = cv2.VideoCapture('grupaB1.mpg')
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

times = []
counted = []

# TODO do zrobienia plik konfiguracyjny jeśli wyjdzie więcej takich zmiennych
percent = 0.2

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
    boxes, weights = hog.detectMultiScale(frame, winStride=(16, 16), hitThreshold=0.5)

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    boxes = non_max_suppression(boxes, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in boxes:
        x = int(xA + (xB - xA) / 2)
        y = int(yA + (yB - yA) / 2)
        found = False

        for person in people:
            if person.check_if_its_me(x, y, frame[x][y]):
                person.set_position(x, y)
                person.set_dev((xB - xA) * percent, (yB - yA) * percent)
                found = True
                break

        if not found:
            try:
                people.append(Person(x, y, (xB - xA) * percent, (yB - yA) * percent, frame[x][y]))
            except IndexError:
                pass

        cv2.circle(frame, (x, y), 5, (0, 0, 255))
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0))
        counter += 1

    for person in people:
        # if not person.updated:
        #     person.update_position()
        cv2.circle(frame, (person.x, person.y), 5, (255, 255, 0))
        person.updated = False

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
# for person in people:
#     print(person.color)
