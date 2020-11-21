import numpy as np
import cv2
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


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture('grupaB1.mpg')
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# the output will be written to output.avi
# out = cv2.VideoWriter(
#     'output.avi',
#     cv2.VideoWriter_fourcc(*'MJPG'),
#     15.,
#     (640, 480))

times = []
counted = []

for i in range(video_length - 1):
    start = time()
    counter = 0

    # Capture frame-by-frame
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
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        counter += 1
    end = time()
    times.append(end - start)
    counted.append(counter)
    # Write the output video
    # out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
# out.release()
# finally, close the window
cv2.destroyAllWindows()
print(f'Avarage time per frame: {np.mean(times)}s')
print(f'Max people counted in single frame: {np.max(counted)}')