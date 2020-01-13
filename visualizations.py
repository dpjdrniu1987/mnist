# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 16:05:54 2019

@author: Administrator
"""
import matplotlib.pyplot as plt
import numpy as np
def plot_images_labels_prediction(images,
                                  labels,
                                  prediction,
                                  index,#从第几个index开始显示
                                  num=10):
    fig=plt.gcf()
    fig.set_size_inches(10,12)
    if num>25:
        num=25
    for i in range(0,num):
        ax=plt.subplot(5,5,i+1)
        ax.imshow(np.reshape(images[index],(28,28)),cmap='binary')
        title="label="+str(np.argmax(labels[index]))
        if len(prediction)>0:
            title += ",predict="+str(prediction[index])
            
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        index+=1
    plt.show()
       
