import tkinter as tk
from tkinter.filedialog import askopenfilename
import cv2
import yolo4KFC

window = tk.Tk()
# Code to add widgets will go here...
window.geometry('400x200')
#Action listeners
filename = ""

def clicked_bt1():

    global filename
    acceptable_types = [('Pliki wideo', '*.avi;*.mp4;*.mov;*.mpg')]
    filename = askopenfilename(filetypes=acceptable_types)
    l_accept = tk.Label(window, text=filename)
    l_accept.grid(column=2, row=1)

def clicked_bt2():
    yolo4KFC.show(filename)

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


window.mainloop()