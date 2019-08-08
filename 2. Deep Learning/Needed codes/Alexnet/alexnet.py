import tensorflow as tf
import cv2                 
import numpy as np        
import os                  
from random import shuffle 
from tqdm import tqdm  
 
TRAIN_DIR = 'zeft1'
TEST_DIR='zeft2'
tf.reset_default_graph()
#train_ratio=3780
# General parameters of the model
BATCH_SIZE = 120                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
#MOMENTUM = 0.9
#WEIGHT_DECAY = 0.0005
DROPOUT_KEEP_PROB = 0.5
FC_HIDDEN_SIZE = 1024
K_BIAS = 2
N_DEPTH_RADIUS = 5 
ALPHA = 1e-4
BETA = 0.75
standard_dev=0.01
display_step = 1
f1_shape_height=6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
f1_shape_width=6
training_epochs=25
label_len=7560
display_step1=5
# Global dataset dictionary
dataset_dict = {
    "image_height":224,
    "image_width":224,
    "num_channels": 3,
    "num_labels": 2,          
}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

def label_img(img):
    
     word_label = img.split('_')[-1]
     if  (word_label == '1.jpg'): return [1,0]
     elif (word_label == '0.jpg'): return [0,1]
    
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=img/255.
        img=cv2.resize(img,(720,720))
        img=img[0:480,:]
        img = cv2.resize(img, (dataset_dict["image_width"],dataset_dict["image_height"]))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    #np.save('mytrain_data_224.npy', training_data)   ###########
    return training_data 

train_data = create_train_data()

#train_data = np.load('mytrain_data_224.npy')    
train = train_data
images= np.array([i[0] for i in train])#.reshape(-1,dataset_dict["image_size"],dataset_dict["image_size"],3)
label =np.array([i[1] for i in train])
train=[]
train_data=[]
###############################test_data###############################

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
        img = cv2.resize(img, (dataset_dict["image_width"],dataset_dict["image_height"]))
        test_data.append([np.array(img),np.array(label)])
    shuffle(test_data)
    return test_data 

test_data = create_test_data()

#test_data = np.load('mytest_data_224.npy')
test = test_data
images_test= np.array([i[0] for i in test])#.reshape(-1,dataset_dict["image_size"],dataset_dict["image_size"],3)
label_test =np.array([i[1] for i in test])
test=[]
test_data=[]
###########################################################################

        
# Filter shapes for each layer 
conv_filter_shapes = {
    "c1_filter": [11, 11, 3, 96],
    "c2_filter": [5, 5, 96, 256],
    "c3_filter": [3, 3, 256, 384],
    "c4_filter": [3, 3, 384, 384],
    "c5_filter": [3, 3, 384, 256]
}

# Fully connected shapes
fc_connection_shapes = {
    "f1_shape": [f1_shape_height*f1_shape_width*256,FC_HIDDEN_SIZE],
    "f2_shape": [FC_HIDDEN_SIZE, FC_HIDDEN_SIZE],
    "f3_shape": [FC_HIDDEN_SIZE, dataset_dict["num_labels"]]
}

# Weights for each layer
conv_weights = {
    "c1_weights": tf.Variable(tf.truncated_normal(conv_filter_shapes["c1_filter"],stddev=standard_dev), name="c1_weights"),
    "c2_weights": tf.Variable(tf.truncated_normal(conv_filter_shapes["c2_filter"],stddev=standard_dev), name="c2_weights"),
    "c3_weights": tf.Variable(tf.truncated_normal(conv_filter_shapes["c3_filter"],stddev=standard_dev), name="c3_weights"),
    "c4_weights": tf.Variable(tf.truncated_normal(conv_filter_shapes["c4_filter"],stddev=standard_dev), name="c4_weights"),
    "c5_weights": tf.Variable(tf.truncated_normal(conv_filter_shapes["c5_filter"],stddev=standard_dev), name="c5_weights"),
    "f1_weights": tf.Variable(tf.truncated_normal(fc_connection_shapes["f1_shape"],stddev=standard_dev), name="f1_weights"),
    "f2_weights": tf.Variable(tf.truncated_normal(fc_connection_shapes["f2_shape"],stddev=standard_dev), name="f2_weights"),
    "f3_weights": tf.Variable(tf.truncated_normal(fc_connection_shapes["f3_shape"],stddev=standard_dev), name="f3_weights")
}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
# Biases for each layer
conv_biases = {
    "c1_biases": tf.Variable(tf.truncated_normal([(conv_filter_shapes["c1_filter"][3])],stddev=standard_dev), name="c1_biases"),
    "c2_biases": tf.Variable(tf.truncated_normal([(conv_filter_shapes["c2_filter"][3])],stddev=standard_dev), name="c2_biases"), 
    "c3_biases": tf.Variable(tf.truncated_normal([(conv_filter_shapes["c3_filter"][3])],stddev=standard_dev), name="c3_biases"),
    "c4_biases": tf.Variable(tf.truncated_normal([(conv_filter_shapes["c4_filter"][3])],stddev=standard_dev), name="c4_biases"),
    "c5_biases": tf.Variable(tf.truncated_normal([(conv_filter_shapes["c5_filter"][3])],stddev=standard_dev), name="c5_biases"),
    "f1_biases": tf.Variable(tf.truncated_normal([(fc_connection_shapes["f1_shape"][1])],stddev=standard_dev), name="f1_biases"),
    "f2_biases": tf.Variable(tf.truncated_normal([(fc_connection_shapes["f2_shape"][1])],stddev=standard_dev), name="f2_biases"),
    "f3_biases": tf.Variable(tf.truncated_normal([(fc_connection_shapes["f3_shape"][1])],stddev=standard_dev), name="f3_biases")
}

dataset_dict["total_image_size"] = dataset_dict[ "image_height"] * dataset_dict["image_width"]

# Declare the input and output placeholders
#input_img = tf.placeholder(tf.float32, shape=[BATCH_SIZE, dataset_dict["image_size"], dataset_dict["image_size"], dataset_dict["num_channels"]])
#img_4d_shaped = tf.reshape(input_img, [-1, dataset_dict["image_size"], dataset_dict["image_size"], dataset_dict["num_channels"]])
img_4d_shaped = tf.placeholder(tf.float32, [None,dataset_dict["image_height"],dataset_dict["image_width"],3], name="img_4d_shaped") 
#labels_ = tf.placeholder(tf.float32, shape=[None, dataset_dict["num_labels"]])
labels_= tf.placeholder(tf.float32, [None,dataset_dict["num_labels"]], name="labels_")
keep_prob = tf.placeholder(tf.float32, name="keep_prob") # dropout (keep probability)

# Convolution Layer 1 | Response Normalization | Max Pooling | ReLU
c_layer_1 = tf.nn.conv2d(img_4d_shaped, conv_weights["c1_weights"], strides=[1, 4, 4, 1], padding="SAME", name="c_layer_1")
c_layer_1 += conv_biases["c1_biases"]
c_layer_1 = tf.nn.relu(c_layer_1)
c_layer_1 = tf.nn.lrn(c_layer_1, depth_radius=N_DEPTH_RADIUS, bias=K_BIAS, alpha=ALPHA, beta=BETA)
c_layer_1 = tf.nn.max_pool(c_layer_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

# Convolution Layer 2 | Response Normalization | Max Pooling | ReLU
c_layer_2 = tf.nn.conv2d(c_layer_1, conv_weights["c2_weights"], strides=[1, 1, 1, 1], padding="SAME", name="c_layer_2")
c_layer_2 += conv_biases["c2_biases"]
c_layer_2 = tf.nn.relu(c_layer_2)
c_layer_2 = tf.nn.lrn(c_layer_2, depth_radius=5, bias=K_BIAS, alpha=ALPHA, beta=BETA)
c_layer_2 = tf.nn.max_pool(c_layer_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

# Convolution Layer 3 | ReLU
c_layer_3 = tf.nn.conv2d(c_layer_2, conv_weights["c3_weights"], strides=[1, 1, 1, 1], padding="SAME", name="c_layer_3")
c_layer_3 += conv_biases["c3_biases"]
c_layer_3 = tf.nn.relu(c_layer_3)

# Convolution Layer 4 | ReLU
c_layer_4 = tf.nn.conv2d(c_layer_3, conv_weights["c4_weights"], strides=[1, 1, 1, 1], padding="SAME", name="c_layer_4")
c_layer_4 += conv_biases["c4_biases"]
c_layer_4 = tf.nn.relu(c_layer_4)

# Convolution Layer 5 | ReLU | Max Pooling
c_layer_5 = tf.nn.conv2d(c_layer_4, conv_weights["c5_weights"], strides=[1, 1, 1, 1], padding="SAME", name="c_layer_5")
c_layer_5 += conv_biases["c5_biases"]
c_layer_5 = tf.nn.relu(c_layer_5)
c_layer_5 = tf.nn.max_pool(c_layer_5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

# Flatten the multi-dimensional outputs to feed fully connected layers
#feature_map = tf.reshape(c_layer_5, [-1, 13, 13, 256])
feature_map = tf.reshape(c_layer_5, [-1,f1_shape_height* f1_shape_width* 256],name="feature_map")

# Fully Connected Layer 1 | Dropout
fc_layer_1 = tf.matmul(feature_map, conv_weights["f1_weights"],name="fc_layer_1") + conv_biases["f1_biases"]
fc_layer_1 = tf.nn.relu(fc_layer_1)
fc_layer_1 = tf.nn.dropout(fc_layer_1, keep_prob=DROPOUT_KEEP_PROB)

# Fully Connected Layer 2 | Dropout
fc_layer_2 = tf.matmul(fc_layer_1, conv_weights["f2_weights"], name="fc_layer_2") + conv_biases["f2_biases"]
fc_layer_2 = tf.nn.relu(fc_layer_2)
fc_layer_2 = tf.nn.dropout(fc_layer_2, keep_prob=DROPOUT_KEEP_PROB)

# Fully Connected Layer 3 | Softmax
fc_layer_3 = tf.matmul(fc_layer_2, conv_weights["f3_weights"], name="fc_layer_3") + conv_biases["f3_biases"]
cnn_output = tf.nn.softmax(fc_layer_3,name="cnn_output")               #####
pred=cnn_output   
pred1=tf.argmax(pred,1,name="pred1")                                   #####
####################################################

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer_3, labels=labels_ ),name="cost")
optimizer = tf.train.AdamOptimizer(learning_rate=ALPHA).minimize(cost, name="optimizer")

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(labels_,1), name="correct_pred" )        ####
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")
##########################################################################


##############################Training##############################################
# Initializing the variables
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    #saver = tf.train.Saver(max_to_keep=0)
    saver = tf.train.Saver()
    for epoch in range(training_epochs):   
        avg_cost = 0.
        avg_acc=0.
        total_batch = int(len(label)/BATCH_SIZE)
        counter=0
    # Loop over all batches
        for i in range(total_batch):
            start=counter
            end=start+BATCH_SIZE
            batch_x=np.array(images[start:end])
            batch_y=np.array(label[start:end])
            # Run optimization op (backprop) and cost op (to get loss value)
            
            sess.run(optimizer, feed_dict={img_4d_shaped: batch_x, labels_ : batch_y, keep_prob: DROPOUT_KEEP_PROB})
            c = sess.run(cost, feed_dict={img_4d_shaped: batch_x,
                                                        labels_: batch_y})
            acc = sess.run(accuracy, feed_dict={img_4d_shaped: batch_x, labels_ : batch_y, keep_prob: 1.})  ####
            if i % display_step1 == 0:
             print("Epoch:", '%01d' % (epoch+1),"batch:", '%01d' % (i+1), "cost=","{:.9f}".format(c) + ", training Accuracy= " + "{:.5f}".format(acc))
            # Compute average loss
            avg_cost += c / total_batch
            # Compute average accuracy
            avg_acc += acc / total_batch
            counter+=BATCH_SIZE
        ## Validation after each epoch:
        val_acc=sess.run(accuracy, feed_dict={img_4d_shaped:images_test, labels_:label_test, keep_prob: 1.}) 
        #saver.save(sess,'/home/dell2/Documents/CAA/saved_model_11*111_new/gamed/my_model15_last')
        ##appending:
        
        if epoch % display_step == 0:
            print("Epoch:", '%01d' % (epoch+1), "avg_cost=","{:.9f}".format(avg_cost) + ", avg_training Accuracy= " + "{:.5f}".format( avg_acc)+ ", avg_validation Accuracy= " + "{:.5f}".format( val_acc))
    saver.save(sess,'/home/dell2/Documents/CAA/saved_model_11*11_18epoch/gamed2/my_model20_last2')
    print("Optimization Finished!")

