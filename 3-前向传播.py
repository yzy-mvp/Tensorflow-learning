# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 13:40:47 2022

@author: swaggy.p
"""
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def load_dataset():
    (x,y),(x_val,y_val) = keras.datasets.mnist.load_data()
    #转换为Tensor，
    x = tf.convert_to_tensor(x,dtype=tf.float32) / 255.
    #转换标签
    x = tf.reshape(x, (-1,28*28))
    
    
    y = tf.convert_to_tensor(y,dtype=tf.int32)
    #标签编码
    y = tf.one_hot(y, depth=10)
    
    #建立数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((x,y)).batch(200)
    return train_dataset

def init_paramaters():
    
    # 每层的张量都需要被优化，故使用 Variable 类型，并使用截断的正太分布初始化权值张量
    # 偏置向量初始化为 0 即可
    w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))
    b1 = tf.Variable(tf.zeros([256]))
    
    w2 = tf.Variable(tf.random.truncated_normal([256,128],stddev=0.1))
    b2 = tf.Variable(tf.zeros([128]))
    
    w3 = tf.Variable(tf.random.truncated_normal([128,10],stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))
    
    return w1,b1,w2,b2,w3,b3

#定义一次迭代的训练过程
def train_epoch(epoch, train_dataset, w1,b1,w2,b2,w3,b3, lr = 0.001):
    for step, (x,y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            #第一层
            #h1 = tf.nn.relu(tf.multiply(x, w1) + b1)
            h1 = tf.nn.relu((x @ w1) + b1)
            #第二层
            #h2 = tf.nn.relu(tf.multiply(h1, w2) + b2)
            h2 = tf.nn.relu((h1 @ w2) + b2)
            
            #输出层
            #out = tf.multiply(h2, w3) + b3
            out = h2 @ w3 + b3
            #计算误差
            loss = tf.reduce_mean(tf.square(y-out))
            
            #自动求解梯度
            grads = tape.gradient(loss, [w1,b1,w2,b2,w3,b3])
            
        #更新梯度,原地更新，保证变量的数据类型不变
        w1.assign_sub(lr* grads[0])
        b1.assign_sub(lr* grads[1])
        w2.assign_sub(lr* grads[2])
        b2.assign_sub(lr* grads[3])
        w3.assign_sub(lr* grads[4])
        b3.assign_sub(lr* grads[5])
        
        
        #每100次迭代打印结果
        if step % 100 == 0:
            print(epoch, step, 'Loss:',loss.numpy())
            
        return loss.numpy()
    
#定义多次迭代的训练过程
def train(epochs):
    losses = []
    #加载数据集
    train_dataset = load_dataset()
    #初始化权值和偏置
    w1,b1,w2,b2,w3,b3 = init_paramaters()
    
    #遍历每一次迭代
    for epoch in range(epochs):
        loss = train_epoch(epoch, train_dataset, w1, b1, w2, b2, w3, b3,lr=0.001)
        losses.append(loss)
        
    #绘制误差曲线
    x = [i for i in range(epochs)]
    plt.plot(x,losses,'g-',marker='s',label='train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
if __name__ == '__main__':
    train(20)
    
        
        
    
    
            
        
            
        
    


    
    
    
    
    
    