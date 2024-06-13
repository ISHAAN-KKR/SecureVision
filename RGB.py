import cv2
import math
import cvzone
from ultralytics import YOLO

cam=cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
model=YOLO('rgb_detect.pt')

class_names=['none','person','bike','car','motor','airplane','bus','train','truck','boat','light','hydrant','sign',
            'parking meter','bench','bird','cat','dog','deer','sheep','cow','elephant','bear','zebra','giraffe',
            'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite',
            'baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup',
            'fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
            'donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote',
            'keyboard','cell phone','microwave','oven','toaster','sink','stroller','rider','scooter','vase',
            'scissors','face','other vehicle','license plate'
        ]

while True:
    _,image=cam.read()
    results=model(image, stream=True)

    for r in results:
        boxes=r.boxes
        for bbox in boxes:
            x1,y1,x2,y2=int(bbox.xyxy[0][0]),int(bbox.xyxy[0][1]),\
                        int(bbox.xyxy[0][2]),int(bbox.xyxy[0][3])
            conf=math.ceil(bbox.conf[0]*100)/100
            clnm=int(bbox.cls[0])
            # if conf > 0.3:
            cv2.rectangle(image,(x1,y1),(x2,y2),(225,0,225),1)
            cvzone.putTextRect(image,f'{conf} {class_names[clnm]}',(max(15,x1),max(15,y1)),
                            0.9,1,(0,255,0),(255, 0, 255),cv2.FONT_HERSHEY_PLAIN,5)

    cv2.imshow('Camera',image)
    cv2.waitKey(1)