import cv2
import detect_face
import tensorflow as tf
import numpy as np
import facenet
from scipy import misc


    
def get_face(image, pnet, rnet, onet, i, emb):
    
    margin=20
    image_size=160
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    detect_multiple_faces=True
    flag=0
    fimg=[]
    bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    bounding_boxes = bounding_boxes[bounding_boxes[:,4]>0.9,:]
    nrof_faces = bounding_boxes.shape[0]
    out=np.zeros([image_size,1,3])
    if i % 10 ==0:
        print(bounding_boxes.shape)
        print(bounding_boxes)
    if nrof_faces>0:
        det = bounding_boxes[:,0:4]
        det_arr = []
        img_size = np.asarray(image.shape)[0:2]
        if nrof_faces>1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                det_arr.append(det[index,:])
        else:
            det_arr.append(np.squeeze(det))
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = image[bb[1]:bb[3],bb[0]:bb[2],:]
            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            out = np.concatenate((out,scaled),axis=1)
            prewhitened = facenet.prewhiten(scaled)
            fimg.append(prewhitened)
            flag=1
    if flag == 1 :
        _, jpeg_face = cv2.imencode('.jpg', out)
        fimg_np=np.stack(fimg)
        fembedding = emb(fimg_np)
        return jpeg_face.tobytes(), fembedding
    else:
        img = np.ones((128,128,3)) #img is the background image which appears when you run the program
        _, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes(),np.zeros((1,1))
        
        
