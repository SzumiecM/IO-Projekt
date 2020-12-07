import tkinter as tk
from tkinter.filedialog import askopenfilename
from materials import main2 as main
import cv2
from time import time
import numpy as np
import os


class Window:

    def __init__(self):
        self.master = tk.Tk()

        self.master.title("Human Detection")
        self.master.geometry('400x200')
        self.master.configure(bg="alice blue")

        self.filename = None
        self.number = 1
        self.output_movie_name = ''

        self.l1 = tk.Label(self.master, text="Let's detect !", bg="CadetBlue4", font='Georgia')
        self.l1.pack(fill=tk.X)
        self.l2 = tk.Label(self.master, text="Choose the files to analyze:", font=('Georgia', 10), bg="alice blue")
        self.l2.pack(pady=(5, 0))
        self.l3 = tk.Button(self.master, text="Browse", command=self.clicked_bt1, fg="blue", font=('Georgia', 10))
        self.l3.pack(pady=0)
        self.l4 = tk.Label(self.master, text="No file selected", fg="grey", font=('Georgia', 8), bg="alice blue")
        self.l4.pack(pady=0)
        self.l5 = tk.Button(self.master, text="Start analyzing", command=self.clicked_bt2, fg="blue",
                            font=('Georgia', 10), state="disabled")
        self.l5.pack(pady=12)
        self.l6 = tk.Label(self.master, text="Analyze not started yet", fg="grey", font=('Georgia', 8), bg="alice blue")
        self.l6.pack(pady=0)

    def clicked_bt1(self):
        acceptable_types = [('Pliki wideo', '*.avi;*.mp4;*.mov;*.mpg')]
        self.filename = askopenfilename(filetypes=acceptable_types)
        if self.filename != '':
            self.l4['text'] = self.filename
            self.l5['state'] = 'normal'

    def clicked_bt2(self):
        # TODO: handle exception when someone close the app during analyzing
        try:
            self.l5['state'] = 'disabled'
            filename_without_path = os.path.basename(self.filename)
            self.output_movie_name = '{}_analyze{}.avi'.format(filename_without_path[:-4], self.number)
            logs = open('{}_logs.txt'.format(self.output_movie_name[:-4]), "w+")
            self.show(self.filename, logs)
            logs.close()
            self.number += 1
        except:
            logs.close()

    # logic moved from tracker.py for better communication with gui components
    def show(self, filename, logs):

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        cv2.startWindowThread()

        cap = cv2.VideoCapture(filename)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        out = cv2.VideoWriter(self.output_movie_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                              (frame_width, frame_height))

        ct = main.CentroidTracker()

        sizes = []
        avg_size = 0

        times = []
        counted = []
        people = []

        net = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg', 'yolov4-tiny.weights')
        LABELS = open('../coco.names').read().strip().split("\n")

        our_confidence = 0.6
        our_threshold = 0.1

        for frame_number in range(1, video_length):
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

            # cv2.imshow('frame', frame)
            out.write(frame)

            end = time()
            times.append(end - start)
            counted.append(counter)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            logs.write(f'{frame_number}. Time from frame number {frame_number}: {end - start}\n')
            logs.write(f'{frame_number}. People counted in frame: {counter}\r\n')

            self.l6['text'] = f'Progress: {"{:.2f}".format(100 * frame_number / video_length)}%'
            self.master.update()

        self.l5['state'] = 'normal'
        self.l6['text'] = 'Analyzing finished'
        self.master.update()
        cap.release()
        cv2.destroyAllWindows()

        logs.write(f'\n\nAvarage time per frame: {np.mean(times)}s\n')
        logs.write(f'Max people counted in single frame: {np.max(counted)}\n')
        logs.write(f'People counted: {len(people)}\n')
        logs.close()

    def run(self):
        self.master.mainloop()


if __name__ == '__main__':
    Window().run()
