#بسم الله الرحمن الرحيم
import numpy as np
import cv2

cap=cv2.VideoCapture(0)
x=cap.isOpened()
count=0

##while True:
#ret,frame=cap.read()
#    
#name="frame_0%d.jpg"%count
#cv2.imwrite(name,frame)
#count+=1
#    #if cv2.waitKey(10):
#       # break
#       
       
c=0      
while True:
 ret,frame=cap.read()
 c+=1 
 name="frame_0%d.jpg"%count
 cv2.imwrite(name,frame)
 count+=1
 if (c>5):
    break
cap.release()

