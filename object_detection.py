import cv2 
import numpy as np

# you can download these files from https://pjreddie.com/darknet/yolo/
net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')

classes = []

with open('coco.names','r') as f:
    classes = f.read().splitlines()

print(classes)

#cap = cv2.VideoCapture('http://192.168.0.4:8080/playlist.m3u')

#webcam 
#cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture('video2.mp4')
#img = cv2.imread('cars.jpg')

while True:
    _, img = cap.read()

    height,width,_ = img.shape


    blob = cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB = True,crop=False)

    net.setInput(blob)

    output_layer_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layer_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    print(len(boxes))

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    print("indexes - " ,indexes)


    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size=(len(boxes),3))
    
    if(len(indexes)>0):
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[1],2))
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
            cv2.putText(img,label+ " " + confidence, (x,y+20),font,2 ,(255,255,255),2)

            

    cv2.imshow('Image',img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()