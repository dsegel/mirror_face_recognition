import argparse
import face_recognition
import imutils
from imutils.video import VideoStream
import pickle
import cv2
import os
import numpy as np
import sys
from datetime import datetime
from gtts import gTTS

print(f"[{datetime.now().strftime('%m/%d/%Y %H:%M:%S')}] libraries loaded...")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help="path to where the face cascade resides")
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
args = vars(ap.parse_args())


def now():
    return(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))


def timestamp():
    return(datetime.now().strftime("%Y%m%d_%H%M%S"))


def curTime():
    return(datetime.now().strftime("%H:%M:%S"))


# load the known faces and embeddings along with OpenCV's Haar cascade for face detection
print(f"[{now()}] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

print(f"[{now()}] loading face detector...")
detector = cv2.CascadeClassifier(args["cascade"])

greeting_date_dict = {
    "Everett": "None",
    "Sophie": "None",
    "Graham": "None",
    "Barbara": "None",
    "Daniel": "None",
    }

# initialize the video stream and allow the camera sensor to warm up
print(f"[{now()}] starting video stream...")
# VideoStream is threaded
vs = VideoStream().start()

# regular opencv class
# vs = cv2.VideoCapture(0)
# vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                     for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def speak(message):
    tts = gTTS(text=message, lang='en')
    tts.save("audio.mp3")
    os.system("/usr/bin/mpg321 -q  audio.mp3")


def getDayPart():
    if int(curHour) > 17:
        daypart = 'evening'
    elif int(curHour) > 11:
        daypart = 'afternoon'
    else:
        daypart = 'morning'
    return daypart


# loop over frames from the video file stream
while True:
    # global frame

    frame = vs.read()
    # if ret is False:
    #     print(f"[{now()}] ret is False")
    #     continue

    # print("Got a frame...")
    frame = imutils.resize(frame, width=1024)
    # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    gamma = 1.5
    frame = adjust_gamma(frame, gamma=gamma)

    # convert the input frame from (1) BGR to grayscale (for face
    # detection) and (2) from BGR to RGB (for face recognition)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # print("About to detect")
    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.65,
                                      minNeighbors=6, minSize=(28, 28),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    # if rects is not None:
    #    print(f"[{now()}] Face detected")

    # OpenCV returns bounding box coordinates in (x, y, w, h) order
    # but we need them in (top, right, bottom, left) order, so we
    # need to do a bit of reordering
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)

    # print("      Done getting boxes")
    # loop over the facial embeddings
    for encoding in encodings:
        print(f"[{now()}] found something, now looking for face match...", end="")
        # name = "Unknown"

        # attempt to match each face in the input image to our known encodings
        matches = face_recognition.compare_faces(data["encodings"], encoding)

        # check to see if we have found a match
        if True in matches:
            curHour = datetime.now().strftime("%H")
            curMinute = datetime.now().strftime("%M")
            daypart = getDayPart()
            message = "Good " + daypart + " "

            # find the indexes of all matched faces then initialize a dictionary to count the
            # total number of times each face was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for each recognized face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of votes
            name = max(counts, key=counts.get)

            print(f"identified {name} in the {daypart}")
            filename = name + "_" + timestamp() + ".jpg"
            cv2.imwrite(filename, frame)

            if greeting_date_dict[name] != daypart and name != 'Sophie':
                print(f"[{now()}] No greeting yet, so using: {message}")
                # speak(message)
                greeting_date_dict[name] = daypart
            else:
                print(f"[{now()}] ...already greeted {name} in the {daypart}")
        else:
            print("no matches found")

    if boxes is None:
        cv2.destroyAllWindows()

# do a bit of cleanup and exit
vs.release()
cv2.destroyAllWindows()
vs.stop()
sys.exit(0)
