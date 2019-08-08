import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import matplotlib.pyplot as plt
from tqdm import tqdm     #for loops  
import os                 #to join paths
dest_path='/mnt/CADA98CFDA98B961/Rawan/GP/deep/our dataset/No Accident/non_acc_1'       ##dest path
src_path='/mnt/CADA98CFDA98B961/Rawan/GP/deep/our dataset/No Accident/non_acc_1'        ##src path

########################################################################
'''
## Rotating images:
##needs to be modified for saving#######

def ٌRotate_train_data():
    Rotated_data = []
    for img in tqdm(os.listdir(path)):
        path_ = os.path.join(path,img)
        img = cv2.imread(path_)
        img1=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        rows,cols,channels=img1.shape
        ##Rotation matrix
        R=cv2.getRotationMatrix2D((cols/2,rows/2),90,1)    #usual origin point
        output=cv2.warpAffine(img1,R,(cols,rows))
        Rotated_data.append(np.array( output))
    return Rotated_data
       


'''
#######################################################################

## Flipping images:

def flip_train_data():
    k=177
    for img in tqdm(os.listdir(src_path)):
      path_ = os.path.join(src_path,img)
      img = cv2.imread(path_)
      #img1=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
      output=cv2.flip(img,1)
      #output2=cv2.cvtColor(output,cv2.COLOR_RGB2BGR)
      cv2.imwrite(os.path.join(dest_path , 'random%01i_0.jpg' %k), output)
      k+=1
       


#######################################################################

## Cropping images:

def crop_train_data():
    k=0
    for img in tqdm(os.listdir(src_path)):
       path_ = os.path.join(src_path,img)
       img = cv2.imread(path_)
       #img1=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
       output=img[300:,:]
       output1=output[0:350,:]
       #output2=cv2.cvtColor(output1,cv2.COLOR_RGB2BGR)
       cv2.imwrite(os.path.join(dest_path , 'image%01i.jpg' %k), output1)
       k+=1
       


####################################################################

## Renaming images:
def rename_images():
  k=0
  for img in tqdm(os.listdir(src_path)):
    path_ = os.path.join(src_path,img)
    img = cv2.imread(path_)
    #img1=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    ##Flipping:
    #output2=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(dest_path , 'r1%01i_0.jpg' %k), img)
    k+=1
       

#####################################################################

def main():
    #flip_train_data()
    rename_images()
###############################################################
if __name__=='__main__':
   main()
