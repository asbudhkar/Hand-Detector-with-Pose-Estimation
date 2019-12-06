# Hand-Detector-with-Pose-Estimation  
Code to recognize objects in people's hands using pose estimation for hand detection and VGG classifier  

## Open Pose, a human body keypoint estimator, was used for creating hand detector

## Method:  
The idea is to use the orientation of hand with keypoints corresponding to shoulder, elbow and wrist of hand to predict the bounding box around hand and use the image within bounding box to predict the category of object in hand. Even if the segmentation technique fails due to obstruction or intersecting objects, the hand keypoints which are visible can lead to assist in identification of object in hand.  

## a. Create dataset for training  
1. Videos of person holding objects were collected to use for detecting objects in hand. Videos were captured with cameras at different angles. Dataset included frames extracted from videos captured and for every frame an annotation text file including the name of sample, class and four bounding box coordinates.  

2. CMU Open Pose model was used to capture keypoints for every frame of video. For creating training dataset, OpenCV was used to visualize the keypoints on the frame and captured the shoulder, elbow and wrist coordinates of hand which holds the object. The frames where hand keypoint coordinates were missing were ignored.  

3. Thus, the dataset generated is 3 pairs (x, y) of hand coordinates (shoulder, elbow, wrist) as input to model and bounding box coordinates: bottom left and top right that is 2 – (x, y) pairs from annotation file corresponding to the frame as the ground truth.  
The dataset was split into 70 % train and 30 % test set.  

## b. Train a model to predict the bounding box coordinates  
A 5-layer neural network is trained to generate the bounding box coordinates for object with 3 pairs of hand coordinates as input from the train dataset.  
Smooth Mean Square Error i.e. Huber loss was used along with Adam optimizer.  
With a learning rate of 0.001, the model was trained for 200 epochs.  
The model is saved and used a hand detector for next steps.  

## c. Train a classifier with the objects under consideration  
Images of objects under consideration are taken and classes assigned.  
A pretrained VGG-8 classifier trained on COCO dataset is finetuned using images collected.  

## d. Testing the model  
The test set is used to generate bounding box coordinates using the saved hand detector model.  
The image within boxes is cropped and classified using a VGG-8 network finetuned on objects under consideration.  
The result is stored as a video with the bounding boxes drawn on the frame along with the class of the object.  

## Dataset

1. Frames of videos captured in Folder 'images' 
2. A text file dictionary for every frame in Folder 'dicts'  
Each file consisting of following  

img/image_sample_name.txt  
image_width image_height  

label  
bounding box - x y width height  

For an image file img.jpg, a dictionary img.txt exists  

3. For testing - Videos  

## Running instructions

1. Refer tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation from https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation  
Follow the instructions to run it

2. Create dataset for pose estimation in folders images and dicts

3. Create dataset for hand detector
python picture_demo.py  
Select left or right hand using arrow keys to save hand keypoint coordinates and bounding box coordinates of object in hand in data.csv file  

4. Train hand detector model
python train.py

5. Test hand detector model
python picture_demo_test.py

6. Train a classifier model using images of objects under consideration

7. Recognize objects in videos
python web_demo.py --classes "obj1,obj2,..."

## Future Work
Improving Open Pose keypoint detection and trying other methods for pose estimation like Deep cut and Alpha Pose can be done in future  
Adding hueristic to detect which hand has object and automatically create dataset  
Research techniques for hand estimation to detect the bounding box with more accuracy  
