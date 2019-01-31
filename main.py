# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 22:00:00 2019

Previously implemented an eigenface recognition by hand, now ready to use the 
scikit-learn library for achieving the same result. Biggest takeaway is that the library
took care of the most tedious part of the implementation, namely writing functions to:
            - splitting the data
            - extract eigenvectors
            - construct facespace
            - project testing faces back to facespace

previously, the hand-written implementation was heavy on the PCA and not so much on the
actual classification. was able to explore this with the SVC from sklearn this time. 
Also good introduction to the gridsearch for parameter optimization and a more robust
definition of recall and precision for quantifying performance.

BUT, haven't figured out yet how to do outlier detection yet. This current code operates
under the assumption that every test face belongs to a known subject in the database. It 
doesn't address the possibility that the test face is a new face that belongs to no known
subject, or the possibility that the test 'face' is not a face at all. This is something
that the original eigenface algorithm was very good at by using the distance to the facespace
as a direct metric. 

@author: jh
"""

from orl_data import ORL
from time import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC


"""function to plot faces and predictions"""
def plotFace(faces,guess,real):
    n_row=4
    n_col=int(len(faces)/4)
    fig,ax=plt.subplots(n_row,n_col)
    k=0
    for i in range(0,n_row):
        for j in range(0,n_col):
            a=ax[i][j]
            a.imshow(X_test_show[k],cmap='gray')
            a.set_xticks(())
            a.set_yticks(())
            titleColor=['red','black'][float(y_pred[k])==float(y_test[k])]
            a.set_title(label='predicted: subject {}\n actual: subject {}'.format(y_pred[k],y_test[k]),fontdict={'fontsize':6},color=titleColor)
            k+=1
    plt.tight_layout()

"""start************************************************************
*****************************************************************"""
t0=time()                               #start timing
path=r'cambridge/s{}/{}.pgm'            #path of the testing data
n_sample=40                             #number of subjects to randomly choose from the database
n_fps=10                                #number of different faces to load for each subject
test_percent=28/n_sample*n_fps          #20 test images, for plotting

orlData=ORL(path,n_sample,n_fps)        #custom class for loading the face data, see orl_data.py for details

X=orlData.data                          #face data, matrix of size (n_sample, total pixel per image flattened)
y=orlData.target                        #face label, integers

img=orlData.image                       #face data, unflattened, 3D matrix of size (n_sample, image height, image width)
h,w=img.shape[1:3]                  

"""start by splitting the data into training and testing sets"""
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_percent/100.0,random_state=42)

"""establishing n principle components from PCA on the training images"""
n_component=2                           #first n eigenfaces. note that n_component is <<len(X_train). see Pentland & Turk paper on reason
pca=PCA(n_component,whiten=True).fit(X_train)   #calling sklearn PCA constructor
#print(pca.explained_variance_)                 #uncomment to see eigenvalues associated with each eigenface
#print(pca.explained_variance_ratio_)           #uncomment to see relative weight of each eigenvalue
components=pca.components_
eigfaces=components.reshape(n_component,h,w)    #reshape the components back to the (n_component, image height, image width) for plotting if desired

"""projecting both training and testing images onto the subspace
defined by the orthonormal components from PCA"""
X_train_pca=pca.transform(X_train)
X_test_pca=pca.transform(X_test)

"""build multi-class svm classifer from sklearn
first, grid search for optimal hyperparameters"""
param_grid={'C':list(np.linspace(1e3,1e5,5)),'gamma':list(np.linspace(1e-4,0.1,5))}
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),param_grid, cv=5)
clf = clf.fit(X_train_pca, y_train)
#print(clf.best_estimator_)                     #uncomment to see details of the optimal hyperparameters

y_pred=clf.predict(X_test_pca)                  #predict with svm

"""obtain metrics on precision, recall, f1, confusion matrix"""
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

X_test_show=X_test.reshape((len(X_test),h,w))   #reshape X_test images back to (len(X_test), image height, image width) for plotting
plotFace(X_test_show,y_pred,y_test)             #plot faces with predictions

print('total runtime: {}s'.format(round(time()-t0,3)))