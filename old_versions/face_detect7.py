import argparse
import face_recognition
import imutils
import pickle
import time
import cv2
import os
import datetime
import numpy as np
from gtts import gTTS

print("[INFO] Libraries loaded...")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
    help = "path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings...")
#file = open("encodings_cnn_20190902.pickle",'rb')
#data = pickle.load(file)
data = pickle.loads(open(args["encodings"], "rb").read())

print("[INFO] loading face detector...")
detector = cv2.CascadeClassifier(args["cascade"])
#detector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

greeting_date_dict = {
    "Everett": "None",
    "Sophie": "None",
    "Graham": "None",
    "Barbara": "None",
    "Daniel": "None",
    }

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def speak(message):
    tts = gTTS(text=message, lang='en')
    tts.save("audio.mp3")
    os.system("/usr/bin/mpg321 -q  audio.mp3")
    #os.system("echo {} | python3 /home/pi/MagicMirror/modules/MMM-Polly/polly_client.py --lang=en-US -voice Penelope".format(message))

def getDayPart():
    global today, now, daypart
    now = datetime.datetime.now()
    curHour = now.hour
    today = now.strftime("%Y%m%d")
    if curHour > 17:
        daypart = 'evening'
    elif curHour > 11:
        daypart = 'afternoon'
    else:
        daypart = 'morning'
    return daypart

# loop over frames from the video file stream
while True:
    global frame
    now = datetime.datetime.now()
    curHour = now.hour
    curMinute = now.minute
    curSecond = now.second
    myTime = now.strftime("%Y%m%d_%H%M%S")
    daypart = getDayPart()
    message = "Good " + daypart + " "
    #message = "Good+" + daypart + "+"

    ret, frame = vs.read()
    frame = imutils.resize(frame, width=1024)
    #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    gamma = 1.5
    frame = adjust_gamma(frame, gamma=gamma)
    
    # convert the input frame from (1) BGR to grayscale (for face
    # detection) and (2) from BGR to RGB (for face recognition)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #print("About to detect")
    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.65, 
        minNeighbors=6, minSize=(28, 28),
        flags=cv2.CASCADE_SCALE_IMAGE)
    #if rects is not None:
    #    print("Face detected")

    # OpenCV returns bounding box coordinates in (x, y, w, h) order
    # but we need them in (top, right, bottom, left) order, so we
    # need to do a bit of reordering
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    #print("      Done getting boxes")
    # loop over the facial embeddings
    for encoding in encodings:
        print("Looking for face match")
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
            encoding)
     #      print("Found match")

        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)

            print("[INFO][{}] Identified: {} during hour {}, {}".format(myTime, name, now.hour, daypart))
            filename = name + "_" + myTime + ".jpg"
            cv2.imwrite(filename, frame)

    #        filename_box = name + "_" + time + "_box.jpg"
    #        for (top, right, bottom, left) in boxes:
    #            # draw the predicted face name on the image
    #            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    #            y = top - 15 if top - 15 > 15 else top + 15
    #            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    #            cv2.imwrite(filename_box, frame)
            
            if name in ['Barbara', 'Daniel', 'Everett'] and greeting_date_dict[name] != daypart:
                greeting_date_dict[name] = daypart
                message = "Good " + daypart + ", " + name
                speak(message)
            elif name == 'Graham' and greeting_date_dict[name] != daypart:
                greeting_date_dict[name] = daypart
                message = "I'm watching you, Graham"
                speak(message)
            elif name == 'Sophie' and greeting_date_dict[name] != now.hour:
                poem = open("/home/pi/poem.txt","r")
                lines = poem.read()
                tts = gTTS(text=lines, lang='en')
                tts.save("poem.mp3")
                greeting_date_dict[name] = now.hour
                os.system("/usr/bin/mpg321 -q  meow.mp3")
                message = "Good " + daypart + ", " + name
                speak(message)
                message = "Would you like to hear a poem?"
                speak(message)
                message = "OK. Here goes."
                speak(message)
                os.system("/usr/bin/mpg321 -q  poem.mp3")
                time.sleep(1)
                speak("Sorry, I forget the rest.")

            filename = name + "_" + myTime + ".jpg"

# do a bit of cleanup and exit
vs.release()
cv2.destroyAllWindows()
vs.stop()
