#!/usr/bin/env python
#
# Project: Video Streaming with Flask
# Author: Log0 <im [dot] ckieric [at] gmail [dot] com>
# Date: 2014/12/21
# Website: http://www.chioka.in/
# Description:
# Modified to support streaming out with webcams, and not just raw JPEGs.
# Most of the code credits to Miguel Grinberg, except that I made a small tweak. Thanks!
# Credits: http://blog.miguelgrinberg.com/post/video-streaming-with-flask
#
# Usage:
# 1. Install Python dependencies: cv2, flask. (wish that pip install works like a charm)
# 2. Run "python main.py".
# 3. Navigate the browser to the local webpage.
from flask import Flask, render_template, Response
from camera import VideoCamera
import detect_face
import tensorflow as tf
import get_face
import numpy as np
import cv2
import facenet

app = Flask(__name__)

#pnet, rnet, onet
frame = np.ones([512,512,3])

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    global frame
    while True:
        frame = camera.get_frame()
        _, capture = cv2.imencode('.jpg', frame)
        capture=capture.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + capture + b'\r\n\r\n')

def gen2():
    i = 0
    while True:
        face,a = get_face.get_face(frame,pnet,rnet,onet,i,emb)
        i=i+1
        print(a.shape)
        nrof_images=a.shape[0]
        sameperson=np.zeros((nrof_images,nrof_images))
        # Print distance matrix
        if nrof_images > 1 :
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(a[i,:], a[j,:]))))
                    if i is not j and dist<=1.0 :
                        sameperson[i,j]=1
                    print('  %1.4f  ' % dist, end='')
                print('')
            print('sameperson?')
            print(sameperson)
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + face + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_face')
def video_feed_face():
    return Response(gen2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    global pnet, rnet, onet
    global emb
    
    mtcnn_graph=tf.Graph()
    with mtcnn_graph.as_default() :
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        pnet,rnet,onet = detect_face.create_mtcnn(sess,None)
        
    facenet_graph = tf.Graph()
    with facenet_graph.as_default() :
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        sess_facenet = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess_facenet.as_default() :
            facenet.load_model('models/20180402-114759')
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings=tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            emb = lambda images : sess_facenet.run(embeddings,feed_dict={images_placeholder: images, phase_train_placeholder:False})
    '''
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    '''
    app.run(host='0.0.0.0', debug=True)
    
