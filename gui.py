import tkinter as tk
from tkinter.filedialog import askopenfilename
import cv2
import inport as inp
window = tk.Tk()
# Code to add widgets will go here...
window.geometry('400x200')
#Action listeners
filename = ""

def clicked_bt1():

    global filename, vidcap
    acceptable_types = [('Pliki wideo', '*.avi;*.mp4;*.mov')]
    filename = askopenfilename(filetypes=acceptable_types)
    vidcap = cv2.VideoCapture(filename)
    l_accept = tk.Label(window, text=filename)
    l_accept.grid(column=2, row=1)

def clicked_bt2():
    if filename != "":
        print(filename)
        inp.ifSuccess(vidcap)
#Appereance
l_vid_choice = tk.Label(window, text="Wybierz nagranie\n do analizy")
l_vid_choice.grid(column=0, row=1)
bt_vid_choice = tk.Button(window, text="Przeglądaj", command=clicked_bt1, fg="blue")
bt_vid_choice.grid(column=1, row=1)
l_accept = tk.Label(window, text="Brak wgranego pliku", fg="grey")
l_accept.grid(column=1, row=2)
l_vid_choice = tk.Label(window, text="Rozpocznij analizę\n wybranego nagrania")
l_vid_choice.grid(column=0, row=3)
bt_an_start = tk.Button(window, text="Start analizy", command=clicked_bt2, fg="red")
bt_an_start.grid(column=1, row=3)
l_vid_choice = tk.Label(window, text="Zapisz film\n wynikowy po analizie")
l_vid_choice.grid(column=0, row=4)
bt_save_vid = tk.Button(window, text="Zapisz film", command=clicked_bt1, fg="blue")
bt_save_vid.grid(column=1, row=4)
l_vid_choice = tk.Label(window, text="Odtwórz przeanalizowane\n nagranie")
l_vid_choice.grid(column=0, row=5)
bt_save_vid = tk.Button(window, text="Odtwórz film", command=clicked_bt1, fg="blue")
bt_save_vid.grid(column=1, row=5)

window.mainloop()