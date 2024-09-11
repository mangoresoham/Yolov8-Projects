from numpy import source
from ultralytics import YOLO
import cv2
import cvzone
import math



cap=cv2.VideoCapture(0)    #For Web Cam
cap.set(3,1080) #width
cap.set(4,720)  #height 


# cap=cv2.VideoCapture('D:\Programming\Projects\YOLO\Webcam\Videos\motorbikes-1.mp4')     #For Video
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

while True:
    success, img=cap.read()
    '''By setting stream=True, the YOLO model processes each frame of the video stream as soon as it's captured,
        enabling real-time object detection. '''
    results=model(img,stream=True)

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
            cvzone.cornerRect(img,(x1,y1,w,h))  

            # CONFIDENCE SCORE
            '''ceil the confidence score to get upto 2 decimals'''
            conf=math.ceil((box.conf[0]*100))/100
            '''display the score on rectangle'''
            # cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1))) #use max function to keep the rectangle amd text of confidence within the frame

            #CLASS NAME
            cls =int(box.cls[0])
            cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=1,thickness=1)


    cv2.imshow("Image",img)
    cv2.waitKey(1)