from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
import os
import tkinter as tk
from PIL import Image,ImageTk
from tkinter.filedialog import askopenfile
import cv2
import numpy as np
import winsound

root=tk.Tk()
root.title("Object Detection and Identification")
root.configure(background='#f0f0f0')

canvas=tk.Canvas(root,width=600,height=400)
canvas.grid(columnspan=3)

#logo
logo=Image.open('final.png')
logo=ImageTk.PhotoImage(logo)
logo_label=tk.Label(image=logo)
logo_label.image=logo
logo_label.grid(column=1,row=0)

#instruction
instructions=tk.Label(root,text="                           Choose a Option",font="Raleway",fg='#f45221')
instructions.grid(columnspan=3,column=0,row=1)

def open_file():
    browse_text.set("loading...")
    file=askopenfile(parent=root,title="Please select Image/Video")
    return file.name

def object_detection():
    
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    
    classes = []
    with open("coco.names", "r") as f:
        classes = f.read().splitlines()
    cap=cv2.VideoCapture(open_file())
    #cap = cv2.VideoCapture('test2.mp4')
    #cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('image.jpg')
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))
    
    while True:
        _, img = cap.read()
        height, width, _ = img.shape
    
        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)
    
        boxes = []
        confidences = []
        class_ids = []
    
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
    
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    
        if len(indexes)>0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                color = colors[i]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
    
        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
        if key==27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    browse_text.set("Browse")

def webcam_object_detection():
    webcam_text.set("loading...")
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    
    classes = []
    with open("coco.names", "r") as f:
        classes = f.read().splitlines()
    #cap=cv2.VideoCapture(open_file())
    #cap = cv2.VideoCapture('test2.mp4')
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture('image.jpg')
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))
    
    while True:
        _, img = cap.read()
        height, width, _ = img.shape
    
        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)
    
        boxes = []
        confidences = []
        class_ids = []
    
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
    
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    
        if len(indexes)>0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                color = colors[i]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
    
        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
        if key==27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    webcam_text.set("Live Detection")

def mask_detect():
        def detect_and_predict_mask(frame, faceNet, maskNet):
                # grab the dimensions of the frame and then construct a blob
                # from it
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                        (104.0, 177.0, 123.0))

                # pass the blob through the network and obtain the face detections
                faceNet.setInput(blob)
                detections = faceNet.forward()
                print(detections.shape)

                # initialize our list of faces, their corresponding locations,
                # and the list of predictions from our face mask network
                faces = []
                locs = []
                preds = []

                # loop over the detections
                for i in range(0, detections.shape[2]):
                        # extract the confidence (i.e., probability) associated with
                        # the detection
                        confidence = detections[0, 0, i, 2]

                        # filter out weak detections by ensuring the confidence is
                        # greater than the minimum confidence
                        if confidence > 0.5:
                                # compute the (x, y)-coordinates of the bounding box for
                                # the object
                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                (startX, startY, endX, endY) = box.astype("int")

                                # ensure the bounding boxes fall within the dimensions of
                                # the frame
                                (startX, startY) = (max(0, startX), max(0, startY))
                                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                                # extract the face ROI, convert it from BGR to RGB channel
                                # ordering, resize it to 224x224, and preprocess it
                                face = frame[startY:endY, startX:endX]
                                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                                face = cv2.resize(face, (224, 224))
                                face = img_to_array(face)
                                face = preprocess_input(face)

                                # add the face and bounding boxes to their respective
                                # lists
                                faces.append(face)
                                locs.append((startX, startY, endX, endY))

                # only make a predictions if at least one face was detected
                if len(faces) > 0:
                        # for faster inference we'll make batch predictions on *all*
                        # faces at the same time rather than one-by-one predictions
                        # in the above `for` loop
                        faces = np.array(faces, dtype="float32")
                        preds = maskNet.predict(faces, batch_size=32)

                # return a 2-tuple of the face locations and their corresponding
                # locations
                return (locs, preds)

        # load our serialized face detector model from disk
        prototxtPath = r"face_detector\deploy.prototxt"
        weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        # load the face mask detector model from disk
        maskNet = load_model("mask_detector.model")

        # initialize the video stream
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()

        # loop over the frames from the video stream
        while True:
                # grab the frame from the threaded video stream and resize it
                # to have a maximum width of 400 pixels
                frame = vs.read()
                frame = imutils.resize(frame, width=400)

                # detect faces in the frame and determine if they are wearing a
                # face mask or not
                (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

                # loop over the detected face locations and their corresponding
                # locations
                for (box, pred) in zip(locs, preds):
                        # unpack the bounding box and predictions
                        (startX, startY, endX, endY) = box
                        (mask, withoutMask) = pred

                        # determine the class label and color we'll use to draw
                        # the bounding box and text
                        label = "Mask" if mask > withoutMask else "No Mask"
                        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                        # include the probability in the label
                        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                        # display the label and bounding box rectangle on the output
                        # frame
                        cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                # show the output frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                        break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()

def security_cam():
        cam = cv2.VideoCapture(0)
        while cam.isOpened():
            ret, frame1 = cam.read()
            ret, frame2 = cam.read()
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
            for c in contours:
                if cv2.contourArea(c) < 5000:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
                winsound.PlaySound('alert.wav', winsound.SND_ASYNC)
            if cv2.waitKey(10) == ord('q'):
                break
            cv2.imshow('Security Camera', frame1)
        cam.release()
        cv2.destroyAllWindows()

#browse button
browse_text=tk.StringVar()
browse_btn=tk.Button(root,textvariable=browse_text,command=lambda:object_detection(),font="Raleway",bg="#f45221",fg="white",height=2,width=15)
browse_text.set("Browse")
browse_btn.grid(column=0,row=2)

#webcam button
webcam_text=tk.StringVar()
webcam_btn=tk.Button(root,textvariable=webcam_text,command=lambda:webcam_object_detection(),font="Raleway",bg="#fbb041",fg="white",height=2,width=15)
webcam_text.set("Live Detection")
webcam_btn.grid(column=1,row=2)

#facemask button
facemask_text=tk.StringVar()
facemask_btn=tk.Button(root,textvariable=facemask_text,command=lambda:mask_detect(),font="Raleway",bg="#f45221",fg="white",height=2,width=15)
facemask_text.set("Mask Detection")
facemask_btn.grid(column=3,row=2)

#securitycam button
securitycam_text=tk.StringVar()
securitycam_btn=tk.Button(root,textvariable=securitycam_text,command=lambda:security_cam(),font="Raleway",bg="#f45221",fg="white",height=2,width=15)
securitycam_text.set("Security Camera")
securitycam_btn.grid(column=1,row=3)

root.mainloop()
