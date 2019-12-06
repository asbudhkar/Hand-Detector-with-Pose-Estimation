# Code to use the hand detector model and recognize objects in people's hand

import os
import re
import sys
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from lib.config import cfg, update_config
from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender
from lib.pafprocess import pafprocess
from lib.utils.paf_to_pose import paf_to_pose_cpp

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='pose_model.pth', help ='Path to pose estimation model')
parser.add_argument('--classes', default = "hand,background", type=str, help='string with class dictionary')
parser.add_argument('--weight_classifier', default = "vgg_8.pth", type=str, help='Path to classifier model')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# choose device cpu or gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    

# update config file
update_config(cfg, args)

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

# Load hand detector model
net = Net()
net.load_state_dict(torch.load('model.pth'))	
torch.nn.DataParallel(net)
net.float()
net.eval()


# Initialize model
def initialize_model(model_name, num_classes, use_pretrained=True):

    model_ft = None

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "vgg":
        """ VGG19_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classesa

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

# Load model
def load_model(weight_name,model):

    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(weight_name,map_location='cpu'))

    return model

# Classify image 
def classify(imgs,model):

    model.eval()
    pil_imgs = []
    trans = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.4363, 0.5041, 0.5042], [0.2018, 0.2109, 0.2131])])

    pil_imgs.append(trans(Image.fromarray(imgs)))

    X = torch.stack(pil_imgs)
    X = X
    yhat = model(X)
    l_scores,preds = torch.max(yhat,1)

    return l_scores.detach().cpu().numpy(),preds

def draw_humans(npimg, humans, imgcopy=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    centers = {}
    for human in humans:

        # draw point
        for i in range(CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue
            body_part = human.body_parts[i]
            if (body_part.part_idx==2 or body_part.part_idx==3 or body_part.part_idx==4 or body_part.part_idx==5 or body_part.part_idx==6 or body_part.part_idx==7):
                    center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                    centers[i] = center
                    cv2.circle(npimg, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue
            cv2.line(npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

    return npimg,centers

# Load model    
weight_name = args.weight
model = get_model('vgg19')     
model.load_state_dict(torch.load(weight_name))
model.to(device)
model.float()
model.eval()

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



if __name__ == "__main__":
   
  # Test video 
  video_capture = cv2.VideoCapture('cam2.mp4')
  ret, oriImg = video_capture.read() 
  fourcc = cv2.VideoWriter_fourcc(*'MJPG')

  # Read frames from video 
  ret, oriImg = video_capture.read()
  outvideo = cv2.VideoWriter("hand.avi",fourcc,20.0,(int(video_capture.get(3)),int   (video_capture.get(4))))

  # Initialize classifier
  class_model=initialize_model("vgg",4)

  # Load classifier model 
  load_model(args.weight_classifier,class_model)

  classes = args.classes

  # Create list of classes 
  LABELS = classes.split(",")  

  while True:
        l=[] 
        r=[]

        # Capture frame-by-frame
        ret, oriImg = video_capture.read()
       
        tic = time.time()
        
        shape_dst = np.min(oriImg.shape[0:2])

        # Get results of original image

        oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB) 

        # Get results of image    
        with torch.no_grad():
            paf, heatmap, im_scale = get_outputs(oriImg, model,  'rtpose')

        # Get keypoints for each human    
        humans = paf_to_pose_cpp(heatmap, paf, cfg)

        # Get keypoint coordinates for each hand 
        out,centers = draw_humans(oriImg, humans)
        

        # Get keypoint coordinates for left and right hands
        for center,value in centers.items():
			   
                           
          if(center==2 or center==3 or center==4):
		
                val=''.join(re.sub(r"\(*\)*", "", str(value)))+","
                l1=(val.split(','))
        	
                for i in l1:
                   if i!='':  
                     l.append(int(i)) 
          if(center==5 or center==6 or center==7):
		
                val=''.join(re.sub(r"\(*\)*", "", str(value)))+","
                l1=(val.split(','))
        	
                for i in l1:
                   if i!='':  
                     r.append(int(i))       
					
        # Detect left hand               
        if (len(l)==6):         
           x1=l[0]
           y1=l[1]
           x2=l[2]
           y2=l[3]
           x3=l[4]
           y3=l[5]  	

           # Predict left hand bounding box
           (x,y,w,h)=(net(Variable(torch.Tensor([x1, y1, x2, y2, x3, y3]))))
       
           top_left_x = min([x.int().item(),w.int().item()])
           top_left_y = min([y.int().item(),h.int().item()])
           bot_right_x = max([x.int().item(),w.int().item()])
           bot_right_y = max([y.int().item(),h.int().item()])
           
           oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB) 
           
           oriImg = np.asarray(oriImg) 
           
          # Predict left hand bounding box
           oriImg = draw_humans1(oriImg, x, y,abs(w-x),abs(h-y))
             
           oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)

           # Crop image 
           newImg = oriImg[int(top_left_y):int(bot_right_y), int(top_left_x):int(bot_right_x)]
           
           if(newImg.shape[0]!=0): 
              
               # Classify cropped image 
               scores,index=classify(newImg,class_model)
               
               pred_cls=LABELS[index.numpy()[0]]
               
               if (scores[0]>3 and pred_cls!="background"):
                  cv2.rectangle(oriImg, (x+w,y+h), (x+w+2,y+h+2),color=(0, 255, 0), thickness=2) 
                  cv2.putText(oriImg,pred_cls, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),thickness=2)  
 
           oriImg = cv2.cvtColor(oriImg, cv2.COLOR_RGB2BGR)

           # Save video frame with bounding box and label
           outvideo.write(oriImg)      

        # Detect right hand  
        if (len(r)==6):         
           x1=r[0]
           y1=r[1]
           x2=r[2]
           y2=r[3]
           x3=r[4]
           y3=r[5]  

           # Predict right hand bounding box	
           (x,y,w,h)=(net(Variable(torch.Tensor([x1, y1, x2, y2, x3, y3]))))
      
           top_left_x = min([x.int().item(),w.int().item()])
           top_left_y = min([y.int().item(),h.int().item()])
           bot_right_x = max([x.int().item(),w.int().item()])
           bot_right_y = max([y.int().item(),h.int().item()])
         
           oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB) 
           
           oriImg = np.asarray(oriImg) 
           
           # Predict left hand bounding box
           oriImg = draw_humans1(oriImg, x, y,abs(w-x),abs(h-y))
             
           oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB) 
           
           # Crop image
           newImg = oriImg[int(top_left_y):int(bot_right_y), int(top_left_x):int(bot_right_x)]
          
           if(newImg.shape[0]!=0): 
               
               # Classify cropped image  
               scores,index=classify(newImg,class_model)
              
               # Get class
               pred_cls=LABELS[index.numpy()[0]]
             
               if (scores[0]>3 and pred_cls!="background"):
                  cv2.rectangle(oriImg, (x+w,y+h), (x+w+2,y+h+2),color=(0, 255, 0), thickness=2) 
                  cv2.putText(oriImg,pred_cls, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),thickness=2)  
            
           oriImg = cv2.cvtColor(oriImg, cv2.COLOR_RGB2BGR)

           # Save video frame with bounding box and label
           outvideo.write(oriImg)   
                   
        print('time: ', time.time() - tic)

        if cv2.waitKey(1) & 0xFF == 27:
            break

  # When everything is done, release the capture
  video_capture.release()
  cv2.destroyAllWindows()
