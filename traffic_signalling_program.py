import cv2
import argparse
import numpy as np
import sys
import time

classes = "yolov3.txt"
config = "yolov3.cfg"
weights = "yolov3.weights"
image = "dog.jpg"
classe = None
COLORS = None

names = ['1.mp4', '2.mp4', '3.mp4', '4.mp4']
window_titles = ['first', 'second', 'third', 'fourth']  
cap = [cv2.VideoCapture(i) for i in names]
lights_names = ['1','2','3','4']
imagered = cv2.imread('trafficlight-red.jpg')
imagegreen = cv2.imread('trafficlight-green.jpg')
imagered = cv2.resize(imagered, (80, 200))
imagegreen = cv2.resize(imagegreen, (80, 200))
lightsposition = [(10,10),(600,10),(10,450),(600,450)]
frames = [None] * len(names)
gray = [None] * len(names)
ret = [None] * len(names)

    
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    global classe
    global COLORS
    label = str(classe[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def RecognizeImage(image, camera_no):
    print('Camera No',camera_no)
    print(type(image))

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    with open(classes, 'r') as f:
        global classe
        classe = [line.strip() for line in f.readlines()]
    global COLORS
    COLORS = np.random.uniform(0, 255, size=(len(classe), 3))

    net = cv2.dnn.readNet(weights, config)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    count = 0
    objects = ['bicycle', 'car', 'motorcycle', 'bus', 'truck']

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        
        if str(classe[class_ids[i]]) in objects:
            count = count+1
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    
    positions = [(90,10),(700,10),(90,400),(700,400)]
    lightsposition = [(10,10),(600,10),(10,500),(600,500)]
    pos = positions[camera_no]
    camera_no = "Camera "+str(camera_no)
    image = cv2.resize(image, (280, 340))
    cv2.namedWindow(camera_no)
    cv2.moveWindow(camera_no,pos[0],pos[1])
    cv2.imshow(camera_no, image)
    cv2.waitKey(1)
    
    return count


def extractImages(pathIn):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
      vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*5000))    
      success,image = vidcap.read()
      if success:
          RecognizeImage(image)
      print ('Read a new frame: ', success)
      cv2.imwrite("\\frame%d.jpg" % count, image)     
      count = count + 1


def getTrafficCount(frame_position):
    counting = {}
    for i,c in enumerate(cap):
        if c is not None:
            if frame_position==0:
                ret[i], frames[i] = c.read()
            else:
                c.set(cv2.CAP_PROP_POS_MSEC,(frame_position*2000))
                ret[i], frames[i] = c.read()
            
    for i,f in enumerate(frames):
        print(i)
        if ret[i] is True:
            count = RecognizeImage(f,i)
            print(count)
            counting[i] = count
    return counting


def changer(record, traffic_count, current_open):
    print("Changer", current_open)
    newcurrent = -1
    isFull = 0
    for i in range(len(record)):
        if record[i]==0:
            break
        elif i+1==len(record) and record[i]==1:
            record=[0,0,0,0]
            
            
    for k,v in traffic_count.items():
        if record[k]==0:
            newcurrent = k
            record[k]=1
            break
    print("NewCurrent", newcurrent)
    return newcurrent,record,current_open


def changeLights(current_open, old_open):
    cv2.imshow(str(current_open+1), imagegreen)
    cv2.imshow(str(old_open+1), imagered)
    
    
def initTrafficLights(number):
    name = str(number+1)
    pos = lightsposition[number]
    cv2.namedWindow(name)
    cv2.moveWindow(name,pos[0], pos[1])
    cv2.imshow(name, imagered)
    
def main():
    old_open= 0
    for i in range(len(lights_names)):
        initTrafficLights(i)
    current_open = 0
    record = [1,0,0,0]
    timestamp = int(time.time())
    print(timestamp)
    frame_position = 0
    cv2.imshow(lights_names[current_open],imagegreen)
    while True:
        if timestamp+60 < int(time.time()):
            current_open,record,old_open = changer(record, traffic_count, current_open)
            changeLights(current_open,old_open)
            timestamp = int(time.time())
        traffic_count = getTrafficCount(frame_position)
        traffic_count = dict(sorted(traffic_count.items(), key = lambda kv:(kv[1], kv[0]), reverse = True))
        print(traffic_count)
        frame_position= frame_position+1
        if traffic_count[current_open] == 0:
            current_open,record,old_open = changer(record, traffic_count, current_open)
            changeLights(current_open,old_open)
            timestamp = int(time.time())
            print(current_open, record)
        
    return 0

main()
cv2.destroyAllWindows()
