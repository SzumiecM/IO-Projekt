import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import logging


def switch_color(winner):
    return {
        # switch case for one dominant color
        '0': 'blue',
        '1': 'green',
        '2': 'red',
    }[winner]


class CentroidTracker:
    def __init__(self):
        # counter to assign unique IDs to each person
        self.nextObjectID = 1
        # dictionary that stores person ID as the key and the centroid (x,y) coordinates as val
        self.objects = OrderedDict()
        # stores the information about person clothing colour
        self.colours = OrderedDict()
        # stores the information for how long a particular person has ben marked as lost
        self.disappeared = OrderedDict()
        # the number of consecutive frames a person is allowed to be marked as lost/disappeared
        # until we deregister it
        self.maxFramesDisappeared = 40

    def register(self, centroid, box, frame, timestamp):
        # use the next available UID
        self.objects[self.nextObjectID] = centroid
        self.colours[self.nextObjectID] = self.setColour(box, frame)
        bgr = self.setColour(box, frame)
        try:
            if any(max(bgr) - x > 10 for x in bgr):
                color = switch_color(str(bgr.tolist().index(max(bgr))))
            else:
                if np.mean(bgr) < 100:
                    color = 'black'
                elif np.mean(bgr) > 150:
                    color = 'white'
                else:
                    color = 'grey'
        except Exception as e:
            print(e)

        minutes = f'{int(timestamp / 60)}m' if int(timestamp / 60) > 0 else ''
        seconds = f'{round(timestamp % 60)}s'
        logging.info(
            f'Person of ID: {self.nextObjectID} found at: {minutes}{seconds} with center coordinates: {centroid} and average color of clothing: {str(self.colours[self.nextObjectID])} classified as {color}')
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID, timestamp):
        minutes = f'{int(timestamp / 60)}m' if int(timestamp / 60) > 0 else ''
        seconds = f'{round(timestamp % 60)}s'
        logging.info(f'Person of ID: {objectID} disappeared at: {minutes}{seconds}')
        # to deregister person we delete the object ID from both dict
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.colours[objectID]

    def update(self, inputCentroids, boxes, frame, timestamp):
        # rects - list of centerX, centerY coordinates of bounding boxes
        # first - check if it is empty
        if len(inputCentroids) == 0:
            # loop over existing tracked people and mark then as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # deregister the person if it reached maximum number of frames where it
                # has been marked as missing
                if self.disappeared[objectID] > self.maxFramesDisappeared:
                    self.deregister(objectID, timestamp)

            return self.objects

        # create an array of input centroids for the current frame

        # if we are currently not tracking any people take the input centroids
        # and register them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], boxes[i], frame, timestamp)

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
                        self.deregister(objectID, timestamp)

            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], boxes[col], frame, timestamp)
            # return the set of trackable objects
        return self.objects

    def setColour(self, box, frame):
        crop_img = frame[box[1]: int(box[1] + box[3] / 2), box[0]:int(box[0] + box[2])]
        avg_color_per_row = np.average(crop_img, axis=0)
        avg_colors = np.average(avg_color_per_row, axis=0)
        return avg_colors
