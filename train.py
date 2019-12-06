import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import cv2
import sys

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

input = Variable(torch.randn(1,6), requires_grad=True)

out = net(input)

import torch.optim as optim
criterion = torch.nn.SmoothL1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

data=[]
f=open('data.csv', "r")
lines = f.readlines()
for line in lines:
    line=line.rstrip()
    data.append([int(s) for s in line.split(",")])
          

min_loss=sys.maxsize
for epoch in range(100):
    for i, data2 in enumerate(data):
        x1, y1,x2,y2,x3,y3, bx1, by1, bx2, by2 = iter(data2)
        X, Y = Variable(torch.FloatTensor([x1, y1, x2, y2, x3, y3]), requires_grad=True), Variable(torch.FloatTensor([bx1, by1, bx2, by2]), requires_grad=False)

        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if (i!=0 and i % 99 == 0):
            print("Epoch {} - loss: {}".format(epoch, loss.data))
        if(loss<min_loss): 
                min_loss=loss
                torch.save(net.state_dict(), 'model.pth')


(x,y,w,h)=(net(Variable(torch.Tensor([310, 134, 391, 258, 470, 207]))))
print((x,y,w,h))

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

oriImg = cv2.imread("images/sample3_cam2_627.jpg")


out = draw_humans1(oriImg,x,y,abs(w-x),abs(h-y))
cv2.imshow('result.png',out) 
cv2.waitKey(0)
cv2.destroyAllWindows()
