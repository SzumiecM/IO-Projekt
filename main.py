import numpy as np
import cv2
from time import time,sleep
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
    def __init__(self, maxFramesDisappeared=60):  # TODO do zwiększenia??
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
