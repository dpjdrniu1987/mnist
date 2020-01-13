# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 08:22:57 2019

@author: Administrator
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import visualizations as v
from time import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None,784], name="X")
y = tf.placeholder(tf.float32, [None,10], name="Y")

#隐藏层神经元数量
H1_NN=256
W1=tf.Variable(tf.random_normal([784,H1_NN]))
b1=tf.Variable(tf.zeros(([H1_NN])))
Y1=tf.nn.relu(tf.matmul(x,W1)+b1)

W2=tf.Variable(tf.random_normal([H1_NN,10]))
b2=tf.Variable(tf.zeros([10]),name="b")
forward=tf.matmul(Y1,W2)+b2
pred=tf.nn.softmax(forward)

train_epochs=20
batch_size=50
total_batch=int(mnist.train.num_examples/batch_size)
display_step=1
learning_rate=0.01

#oss_function=tf.reduce_mean(tf.square(y-pred))#均方差
#loss_function=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))#交叉熵
loss_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forward,labels=y))

optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

#检查预测类别与实际类别的匹配情况
correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
#准确率，将布尔值转换为浮点数，并计算平均值
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#记录训练开始时间
startTime=time()
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

logdir='d:/tensorflow_log'
sum_loss_op=tf.summary.scalar("loss",loss_function)
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter(logdir,sess.graph)



for epoch in range (train_epochs):
    loss_sum = 0
    for batch in range(total_batch):         
        xs,ys=mnist.train.next_batch(batch_size)      
        _, summary_str,loss=sess.run([optimizer, sum_loss_op,loss_function],feed_dict={x:xs,y:ys})
        writer.add_summary(summary_str,epoch)
        loss_sum = loss_sum + loss
        
        #一代训练完成后，使用验证数据计算误差与准确率；验证集没有分批
        loss,acc=sess.run([loss_function,accuracy],feed_dict={x:mnist.validation.images,y:mnist.validation.labels})
    #打印训练过程中的详细信息
    if (epoch+1)%display_step==0:
        print("Tratin Epoch:",'%02d'%(epoch+1),"loss=","{:.9f}".format(loss),
              "Accuracy=","{:.4f}".format(acc))      
   
#    loss_average=loss_sum/total_batch    
#    print("epoch=",epoch+1,"loss_average=",loss_average)
    
#显示训练总时间
duration=time()-startTime
print("Train Finished takes:","{:.2f}".format(duration))

accu_test=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
print("Test Accuracy:",accu_test)


prediction_result=sess.run(tf.argmax(pred,1),feed_dict={x:mnist.test.images})


v.plot_images_labels_prediction(mnist.test.images,
                              mnist.test.labels,
                              prediction_result,10,10)


