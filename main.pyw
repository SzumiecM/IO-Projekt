import tkinter as tk
from tkinter.filedialog import askopenfilenames
from tracker import CentroidTracker
import cv2
from time import time
import numpy as np
import os
import logging
from warnings import filterwarnings
from datetime import datetime

# ignore numpy warnings, we know what we're doing
filterwarnings('ignore')


class Window:

    def __init__(self):
        self.master = tk.Tk()
        self.master.resizable(False, False)
        self.master.title("Clienter")
        self.master.geometry('400x200')
        self.master.configure(bg="alice blue")

        self.filenames = None

        self.l1 = tk.Label(self.master, text="Let's detect !", bg="CadetBlue4", font='Georgia')
        self.l1.pack(fill=tk.X)
        self.l2 = tk.Label(self.master, text="Choose one or more files to analyze:", font=('Georgia', 10), bg="alice blue")
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
        self.l7 = tk.Button(self.master, text="Output Videos", command=self.open_videos, fg="blue", font=('Georgia', 10))
        self.l7.pack(side=tk.LEFT)
        self.l8 = tk.Button(self.master, text="Output Logs", command=self.open_logs, fg="blue", font=('Georgia', 10))
        self.l8.pack(side=tk.RIGHT)

        if not os.path.exists(f'{os.getcwd()}/output_logs'):
            os.makedirs(f'{os.getcwd()}/output_logs')
        if not os.path.exists(f'{os.getcwd()}/output_videos'):
            os.makedirs(f'{os.getcwd()}/output_videos')

    def open_videos(self):  # opens folder with output videos
        os.startfile(f'{os.getcwd()}/output_videos')

    def open_logs(self):    # opens folder with output logs
        os.startfile(f'{os.getcwd()}/output_logs')

    def clicked_bt1(self):  # opens explorer for user to choose input files from
        acceptable_types = [('Pliki wideo', '*.avi;*.mp4;*.mov;*.mpg')]
        self.filenames = askopenfilenames(filetypes=acceptable_types)
        if self.filenames != '' and len(self.filenames) == 1:
            self.l4['text'] = self.filenames[0]
            self.l5['state'] = 'normal'
        elif len(self.filenames) > 1:
            self.l4['text'] = 'Multiple files chosen'
            self.l5['state'] = 'normal'

    def clicked_bt2(self):  # starts the analyze
        # handle exception when someone close the app during analyzing
        try:
            self.l5['state'] = 'disabled'
            for i in range(len(self.filenames)):
                filename_without_path = os.path.basename(self.filenames[i])
                log = logging.getLogger()  # root logger
                for hdlr in log.handlers[:]:  # remove all old handlers
                    log.removeHandler(hdlr)
                logging.basicConfig(filename=f'output_logs/{filename_without_path}.log', filemode='w',
                                    level=logging.INFO, format=f'{filename_without_path}: %(message)s')
                logging.info(datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
                self.show(self.filenames[i], filename_without_path, i + 1, len(self.filenames))
        except tk.TclError:
            print('Application unexpectedly closed, current progress saved.')

    def show(self, filename, filename_without_path, file_number, files_total):  # core function used for detection
        cv2.startWindowThread()

        cap = cv2.VideoCapture(filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length_in_seconds = video_length / fps

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        out = cv2.VideoWriter(f'output_videos/{filename_without_path}_analyzed.avi',
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))
        # initialize centroid tracker class
        ct = CentroidTracker()

        times = []

        net = cv2.dnn.readNetFromDarknet('dependencies/yolov4-tiny.cfg', 'dependencies/yolov4-tiny.weights')
        LABELS = open('dependencies/coco.names').read().strip().split("\n")

        our_confidence = 0.6
        our_threshold = 0.1

        for current_frame in range(1, video_length):

            counter = 0
            start = time()
            ret, frame = cap.read()
            raw_frame = frame.copy()
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
            NMSboxes = []

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
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
            # non maximum suppresion to remove overlapping boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, our_confidence,
                                    our_threshold)

            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    NMSboxes.append((x, y, w, h))
                    centroids.append((boxes[i][0] + boxes[i][2] / 2, boxes[i][1] + boxes[i][3] / 2))
                    # draw a bounding box rectangle and label on the image
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    counter += 1
            # add new people, update existing, delete vanished
            objects = ct.update(centroids, NMSboxes, raw_frame,
                                timestamp=current_frame * video_length_in_seconds / video_length)

            for (objectID, centroid) in objects.items():
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (int(centroid[0]) - 10, int(centroid[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 4, (0, 255, 0), -1)

            cv2.putText(frame, f'{len(objects)} people currently in frame', (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

            out.write(frame)

            end = time()
            times.append(end - start)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.l6['text'] = f'File {file_number}/{files_total} progress: {"{:.2f}".format(100 * current_frame / video_length)}%'
            self.master.update()

        self.l5['state'] = 'normal'
        self.l6['text'] = 'Analyzing finished'
        self.master.update()
        cap.release()
        cv2.destroyAllWindows()

        logging.info(f'Avarage time for calculations per frame: {"{:.2f}".format(np.mean(times))}s --> {"{:.2f}".format(1/np.mean(times))} fps')

    def run(self):
        self.master.mainloop()


if __name__ == '__main__':
    Window().run()
