import tensorflow as tf
import cv2
import pylab as plt
import numpy as np
import glob
import atexit
print(glob.glob("./*"))
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))   

saver = tf.train.import_meta_graph('./tldetector.tf-1000.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()
    
    

tf_data = graph.get_tensor_by_name("image_input:0")
tf_train = graph.get_tensor_by_name("train_input:0")
tf_label = graph.get_tensor_by_name("labels_input:0")    
sm = graph.get_tensor_by_name("output:0")

def cleanup():
    global sess
    sess.close()
atexit.register(cleanup)

    
def augment(raw,boolean, color, dummy):
    xx, yy = np.meshgrid(np.arange(dummy.shape[3]),np.arange(dummy.shape[2]))

    x = (xx[boolean])
    y = (yy[boolean])

    for i,j in zip(x,y):
        cv2.rectangle(raw, (i*8,j*8), (i*8+64,j*8+64), color, 5)

i = 0
def run(testdata):

    
    #print(np.array(testdata).shape)

    testdata = cv2.resize(testdata,(400,300))
    testdata = cv2.cvtColor(testdata, cv2.COLOR_BGR2RGB)

    img = sess.run([sm], feed_dict={tf_data: testdata.reshape([1,300, 400, 3]), tf_train: False})

        
    threshold = 0.99
        
    img = np.array(img)
    #print("Nothing:",np.sum(img[0,0,:,:,0]>threshold))
    #print("Red:",np.sum(img[0,0,:,:,1]>threshold))
    #print("Green:",np.sum(img[0,0,:,:,2]>threshold))
    #print("Yellow:",np.sum(img[0,0,:,:,3]>threshold))
                
    augment(testdata,img[0,0,:,:,1]>threshold,(255,0,0), img)
    augment(testdata,img[0,0,:,:,2]>threshold,(0,255,0), img)
    augment(testdata,img[0,0,:,:,3]>threshold,(255,255,0), img)
    
    testdata = cv2.cvtColor(testdata, cv2.COLOR_BGR2RGB)
    cv2.imshow('image',testdata)
    cv2.waitKey(1)

    return img


#testdata = (cv2.cvtColor(cv2.imread("../../testimages/00200.jpg"), cv2.COLOR_BGR2RGB))
#run(testdata)