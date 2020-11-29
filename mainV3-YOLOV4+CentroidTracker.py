import numpy as np
import cv2
from time import time
from scipy.spatial import distance as dist
from collections import OrderedDict
from imutils.object_detection import non_max_suppression


# TODO  - zmienne wielkości okienek trackerów?
#           - wyciąganie mediany wielkości kwadratów z kilku sekund???
#       - automatyczny dev <---
#       - ZASTOSOWAĆ DRUGI MECHANIZM WYKRYWANIAAA?? <<<---
#           wydaje się być koniecznością, program totalnie wywala się przy example01_mp4 (nie jest konieczny ale
#           fajnie żeby działał jednak zawsze)
#       - dodać licznik osób na obrazie
class CentroidTracker:
    def __init__(self, maxFramesDisappeared=60): #TODO do zwiększenia??
        # counter to assign unique IDs to each person
        self.nextObjectID = 0
        # dictionary that stores person ID as the key and the centroid (x,y) coordinates as val
        self.objects = OrderedDict()
        # stores the information for how long a particular person has ben marked as lost
        self.disappeared = OrderedDict()
        # the number of consecutive frames a person is allowed to be marked as lost/disappeared
        # until we deregister it
        self.maxFramesDisappeared = maxFramesDisappeared

    def register(self, centroid):
        # use the next available UID
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister person we delete the object ID from both dict
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, boxes):
        # rects - list of centerX, centerY coordinates of bounding boxes
        # first - check if it is empty
        if len(boxes) == 0:
            # loop over existing tracked people and mark then as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # deregister the person if it reached maximum number of frames where it
                # has been marked as missing
                if self.disappeared[objectID] > self.maxFramesDisappeared:
                    self.deregister(objectID)

            return self.objects

        # create an array of input centroids for the current frame

        inputCentroids = boxes
        # if we are currently not tracking any people take the input centroids
        # and register them
        if len(self.objects) == 0:
            for i in range(0, len(boxes)):
                self.register(inputCentroids[i])

        # if not, then we are currently tracking objects so we need to
        # try and match the input centroids to existing people centroids
        else:
            # take the set of people IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of people centroids
            # and input centroids, our goal is to match an input centroid to an
            # existing person centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()

            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxFramesDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
            # return the set of trackable objects
        return self.objects


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

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

ct = CentroidTracker()

sizes = []
avg_size = 0

times = []
counted = []
people = []

net = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg', 'yolov4-tiny.weights')
LABELS = open('coco.names').read().strip().split("\n")

our_confidence = 0.6
our_threshold = 0.1

for i in range(video_length - 1):
    counter = 0
    start = time()
    ret, frame = cap.read()
    (H, W) = frame.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    centroids = []

    # # resizing for faster detection
    # frame = cv2.resize(frame, (640, 480))
    # # using a greyscale picture, also for faster detection
    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > our_confidence and LABELS[classID] == 'person':
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])

                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                # centroids.append((centerX,centerY))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, our_confidence,
                            our_threshold)

    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            centroids.append((boxes[i][0] + boxes[i][2] / 2, boxes[i][1] + boxes[i][3] / 2))
            # draw a bounding box rectangle and label on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            counter += 1

    objects = ct.update(centroids)

    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (int(centroid[0]) - 10, int(centroid[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 4, (0, 255, 0), -1)

    cv2.putText(frame, f'1 person' if counter == 1 else f'{counter} people', (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    out.write(frame)

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
