# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 20:34:54 2019

basic class for loading the ORL face database from cambridge. 
trying to imitate the sklearn databset structures

The size of each image is 92x112 pixels, with 256 grey levels per pixel. 
The images are organised in 40 directories (one for each subject), 
which have names of the form sX, where X indicates the subject number (between 1 and 40). 
In each of these directories, there are ten different images of that subject, 
which have names of the form Y.pgm, where Y is the image number for that subject 
(between 1 and 10).

@author: jh
"""

import numpy as np
import cv2 

class ORL:

    def __init__(self,p,n_s,n_fps):
        self.image=[]
        self.data=[]
        self.target=[]
        self.path=p
        self.n_sample=n_s
        self.n_face=n_fps
        self.sampleIndex=np.random.randint(1,41,self.n_sample)
        self.faceIndex=np.random.randint(1,11,self.n_face)
        
        for sample in self.sampleIndex:
            for face in self.faceIndex:
                self.load=self.path.format(sample,face)
                self.temp=cv2.imread(self.load,0)
                self.image.append(self.temp)
                self.data.extend(self.temp.reshape(1,-11))
                self.target.append(sample)
        
        self.image=np.array(self.image)
        self.data=np.array(self.data)
        self.target=np.array(self.target)
                        



