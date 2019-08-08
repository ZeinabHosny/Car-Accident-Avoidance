import time
import tensorflow as tf
import cv2                 
import numpy as np         
import os                  
from random import shuffle 
from tqdm import tqdm          
cap=cv2.VideoCapture(1)
x=cap.isOpened()
count=1
        
        ##while True:
        #ret,frame=cap.read()
        #    
        #name="frame_0%d.jpg"%count
        #cv2.imwrite(name,frame)
        #count+=1
        #    #if cv2.waitKey(10):
        #       # break
        #       
path='simulation_frames'       
c=0      
while True:
 ret,frame=cap.read()
 c+=1 
 name="frame_%d_0.jpg"%count
 cv2.imwrite(os.path.join(path,name),frame)
 count+=1
 time.sleep(1)
 if (c>19):
   break
cap.release()
    ###################test###############################################
TEST_DIR=path
def label_img(img):
            
   word_label = img.split('_')[-1]
   if (word_label == '1.jpeg') or (word_label == '1.jpg'): return [1,0]
   elif (word_label == '0.jpeg') or (word_label == '0.jpg'): return [0,1]
        
def create_test_data():
    test_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        label = label_img(img)
        path = os.path.join(TEST_DIR,img)
        img = cv2.imread(path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=img/255.
        img=cv2.resize(img,(720,720))
        img=img[0:480,:]
        img = cv2.resize(img, (224,224))
        test_data.append([np.array(img),np.array(label)])
    shuffle(test_data)
           # np.save('mytest_data_actual.npy', test_data)
    return test_data 
            
test_data = create_test_data()
test = test_data
images_test= np.array([i[0] for i in test])#.reshape(-1,dataset_dict["image_size"],dataset_dict["image_size"],3)
label_test =np.array([i[1] for i in test])
test=[]
test_data=[]
        
with tf.Session() as sess:    
            saver = tf.train.import_meta_graph('my_model18_last.meta')
           # saver.restore(sess,tf.train.latest_checkpoint('checkpoint'))
            saver.restore(sess,'my_model18_last')
            graph = tf.get_default_graph()
            img_4d_shaped = graph.get_tensor_by_name("img_4d_shaped:0")
            labels_ = graph.get_tensor_by_name("labels_:0")
            keep_prob = graph.get_tensor_by_name("keep_prob:0")
           
        #    feed_dict ={img_4d_shaped:images_test, labels_:label_test, keep_prob: 1.}
            accuracy = graph.get_tensor_by_name("accuracy:0")
            pred = graph.get_tensor_by_name("correct_pred:0")
            pred_argmax=graph.get_tensor_by_name("pred1:0")
            val_acc,pred_c,pred_arg1=sess.run([accuracy,pred,pred_argmax], feed_dict={img_4d_shaped:images_test, labels_:label_test, keep_prob: 1.}) 
            
            #print ("Testing Accuracy:",val_acc)
            
            ##writing in text file:
            length=len(pred_arg1)
            ss=str(pred_arg1)
            import re
            out=re.findall("\d+",ss)
            file1=open("output.txt","w")
            for i in range(length):
              file1.write(out[i]+'\n')
            file1.close()
            print("end")