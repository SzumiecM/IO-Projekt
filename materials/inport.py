import cv2
def getFrame(sec, count, vidcap):

    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("C:/Users/MrM/Documents/frames/image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames

def ifSuccess(vidcap):
    count = 1
    sec = 0
    frameRate = 0.2 #//it will capture image in each 0.5 second
    success = getFrame(sec, count, vidcap)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec,count, vidcap)

