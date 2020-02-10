#!/usr/bin/python3

# Collision Avoidance - Live Demo

import torch
import torchvision
model = torchvision.models.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
model.load_state_dict(torch.load('/home/jetbot/Notebooks/collision_avoidance/best_model.pth'))
device = torch.device('cuda')
model = model.to(device)

#1. Convert from BGR to RGB
#2. Convert from HWC layout to CHW layout
#3. Normalize using same parameters as we did during training (our camera provides values in [0, 255] range and training loaded images in [0, 1] range so we need to scale by 255.0
#4. Transfer the data from CPU memory to GPU memory
#5. Add a batch dimension"

import cv2
import numpy as np
mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])
normalize = torchvision.transforms.Normalize(mean, stdev)

def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x


#Now, let's start and display our camera. We'll also create a slider that will display the\n",

#import traitlets
#from IPython.display import display
#import ipywidgets.widgets as widgets
#from jetbot import Camera, bgr8_to_jpeg
from jetcam.usb_camera import USBCamera

camera = USBCamera(capture_device=0,width=224, height=224)
camera.running = True
#camera = Camera.instance(width=224, height=224)
#image = widgets.Image(format='jpeg', width=224, height=224)
#image = camera.read()


#blocked_slider = widgets.FloatSlider(description='blocked', min=0.0, max=1.0, orientation='vertical')
#camera_link = traitlets.dlink((camera, 'value'), (image, 'value'), transform=bgr8_to_jpeg)
#camera_link = traitlets.dlink((camera, 'value'), (image, 'value'))


#We'll also create our robot instance which we'll need to drive the motors."
from jetbot import Robot
robot = Robot()

# Next, we'll create a function that will get called whenever the camera's value changes.  This function will do the following steps
#1. Pre-process the camera image
#2. Execute the neural network
#3. While the neural network output indicates we're blocked, we'll turn left, otherwise we go forward.

import torch.nn.functional as F
import time

def update(change):
    global blocked_slider, robot
    x = change['new'] 
    x = preprocess(x)
    y = model(x)
    y = F.softmax(y, dim=1)
    prob_blocked = float(y.flatten()[0])
    #blocked_slider.value = prob_blocked
    print (prob_blocked)

    if prob_blocked < 0.5:
        robot.forward(0.4)
        #robot.right(0.5)
        #robot.left(0.4)
    else:
        robot.left(0.4)
    time.sleep(0.001)

update({'new': camera.value})  


#Cool! We've created our neural network execution function, but now we need to attach it to the camera for processing. 
#WARNING: This code will move the robot!! The collision avoidance should work, but the neural


camera.observe(update, names='value')  # this attaches the 'update' function to the 'value' traitlet of our camera"





