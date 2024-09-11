from unittest import result
from numpy import source
from sympy import limit
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


# cap=cv2.VideoCapture(0)    #For Web Cam
# cap.set(3,1080) #width
# cap.set(4,720)  #height 


cap=cv2.VideoCapture('D:\Programming\Projects\YOLO\Videos\cars.mp4')     #For Video
model=YOLO("YoloWeights\yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


mask=cv2.imread('D:\Programming\Projects\YOLO\Project 1 Car Counter\mask.png')

#Tracking
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limits=[400,297,673,297]
totalCount=[]

while True:
    success, img=cap.read()
    imgRegion = cv2.bitwise_and(img,mask)

    '''It is neccessary to put it inside the while loop inorder to get good quality image with every frame.'''
    #Adding image graphics at top left corner
    imgGraphics=cv2.imread("D:\Programming\Projects\YOLO\Project 1 Car Counter\graphics.png",cv2.IMREAD_UNCHANGED)
    img =cvzone.overlayPNG(img ,imgGraphics,(0,0))

    '''By setting stream=True, the YOLO model processes each frame of the video stream as soon as it's captured,
        enabling real-time object detection. '''
    results=model(imgRegion,stream=True)
    detections=np.empty((0,5))

    

    for r in results:
        boxes=r.boxes
        for box in boxes:

            '''(x1, y1) represents the coordinates of the top-left corner, 
                and (x2, y2) represents the coordinates of the bottom-right corner. '''
            #OPENCV BOUNDUNG BOX
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            '''Substract x2 -x1 & y2-y1 to get the width and height.'''
            w,h=x2-x1,y2-y1
             

            # CONFIDENCE SCORE
            '''ceil the confidence score to get upto 2 decimals'''
            conf=math.ceil((box.conf[0]*100))/100
            '''display the score on rectangle'''
            # cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1))) #use max function to keep the rectangle amd text of confidence within the frame

            #CLASS NAME
            cls =int(box.cls[0])
            currentClass=classNames[cls]

            # give label to cars,bikes,truck only, with confidence score >0.3
            if currentClass=='car' or currentClass=='truck' or currentClass=='bus' or currentClass=='motorbike' and conf>0.3:
                # cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=0.6,thickness=1,offset=3)
                # cvzone.cornerRect(img,(x1,y1,w,h),l=8,rt=5)

                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))

    resultsTracker=tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)


    for result in resultsTracker:
        x1,y1,x2,y2,Id=result
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        print(result)
        w,h=x2-x1,y2-y1
        cvzone.cornerRect(img,(x1,y1,w,h),l=8,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img,f'{int(Id)}',(max(0,x1),max(35,y1)),scale=2,thickness=3,offset=3)

        '''To check if the car passed the line, we check if the center of bounding box touches the line'''
        cx,cy=x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limits[0] <cx < limits[2] and limits[1] -15 < cy <limits[1]+15:
            if totalCount.count(Id)==0:
                totalCount.append(Id)
                cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)


    # cvzone.putTextRect(img,f'Count : {len(totalCount)}',(50,50))
    
    cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255,15),8)


    cv2.imshow("Image",img)
    # cv2.imshow("ImageRegion",imgRegion)
    cv2.waitKey(1)