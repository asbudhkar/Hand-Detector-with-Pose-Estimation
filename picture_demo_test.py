# Code to test hand detector

import os
import re
import sys
import cv2
import math
import time
import scipy
import glob  

import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)

# choose device cpu or gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
weight_name = args.weight

root_path = ''

# Load model
model = get_model('vgg19')     
model.load_state_dict(torch.load(weight_name))
model = torch.nn.DataParallel(model).to(device)
model.float()
model.eval()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6,200)
        self.fc2 = nn.Linear(200,100)
        self.fc3 = nn.Linear(100,50)
        self.fc4 = nn.Linear(50,4)  
    def forward(self, x):
        x = F.relu(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))))
        return x


net = Net()

# Load hand detector model
net.load_state_dict(torch.load('model.pth'))
net = torch.nn.DataParallel(net)
net.float()
net.eval()

def draw_humans1(npimg, x, y, w, h, imgcopy=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
            
    cv2.line(npimg, (x,y),(x,y+h),CocoColors[0],4)
    cv2.line(npimg, (x,y+h),(x+w,y+h),CocoColors[1],4)
    cv2.line(npimg, (x+w,y),(x+w,y+h),CocoColors[2],4)
    cv2.line(npimg, (x+w,y),(x,y),CocoColors[3],4)
    return npimg

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

# Path to dictionary files
path = root_path+'dicts/*.txt'   
files=glob.glob(path)  

# For every dictionary file
for file_name in files:
    print(file_name)
    filename=str(file_name.split('/')[-1])
    filename=(filename.rstrip(".txt"))
     
    f=open(file_name, 'r')  
    lines = f.readlines()

    x,y,w,h=([int(s) for s in lines[-1].split()]) 

    test_image = root_path+'images/'+filename+'.jpg'
	
    oriImg = cv2.imread(test_image) # B,G,R order
    shape_dst = np.min(oriImg.shape[0:2])

    # Get results of image
    with torch.no_grad():
        paf, heatmap, im_scale = get_outputs(oriImg, model,  'rtpose')
	
    # Get keypoints for each human  
    humans = paf_to_pose_cpp(heatmap, paf, cfg)

    # Get keypoint coordinates for each hand
    out,centers = draw_humans(oriImg, humans,x,y,w,h)

    f=open("data.csv",'a+')

    #TODO use some hueristic to select the hand automatically
    while True:
       l=[]
       cv2.imshow('result.png',out)   

       # Press right key if obj in right hand 
       if cv2.waitKey(0) & 0xFF == 83:
            for center,value in centers.items():
                if(center==2 or center==3 or center==4):
                    print(''.join(re.sub(r"\(*\)*", "", str(value)))+",")
                    val=''.join(re.sub(r"\(*\)*", "", str(value)))+",";
                    l1=(val.split(','))
        	
                    for i in l1:
                        if i!='':  
                            l.append(int(i))       
                    
       
            if(len(l)==6):
       	        x1=l[0]
       	        y1=l[1]
       	        x2=l[2]
       	        y2=l[3]
       	        x3=l[4]
       	        y3=l[5]  	

                # Predict the bounding box coordinates for hand using the hand detector model  
                (xpred,ypred,wpred,hpred)=(net(Variable(torch.Tensor([x1, y1, x2, y2, x3, y3]))))     
                out1 = draw_humans1(oriImg,xpred,ypred,abs(wpred-xpred),abs(hpred-ypred))
   
       	        cv2.imshow('result1.png',out1) 
       	        cv2.waitKey(0)
       	        cv2.destroyAllWindows()
           
            break

       #Press left key if obj in left hand 
       elif cv2.waitKey(0) & 0xFF == 81:
           for center,value in centers.items():
                if(center==5 or center==6 or center==7):
                     print(''.join(re.sub(r"\(*\)*", "", str(value)))+",")
                     val=''.join(re.sub(r"\(*\)*", "", str(value)))+",";
                     l1=(val.split(','))
        	
                     for i in l1:
                         if i!='':  
                             l.append(int(i)) 
          
           if(len(l)==6):
       	       x1=l[0]
       	       y1=l[1]
       	       x2=l[2]
       	       y2=l[3]
       	       x3=l[4]
       	       y3=l[5]  	
       	
               # Predict the bounding box coordinates for hand using the hand detector model 
               (xpred,ypred,wpred,hpred)=(net(Variable(torch.Tensor([x1, y1, x2, y2, x3, y3]))))
                       
               out1 = draw_humans1(oriImg,xpred,ypred,abs(wpred-xpred),abs(hpred-ypred))
      
               cv2.imshow('result1.png',out1) 
               cv2.waitKey(0)
       	       cv2.destroyAllWindows()
          
           break

       # Skip if keypoints not detected properly by OpenPose
       elif cv2.waitKey(0) & 0xFF == 27:
           break;    
    f.close()
