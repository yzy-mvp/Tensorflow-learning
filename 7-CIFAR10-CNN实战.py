# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:32:18 2022

@author: swaggy.p
"""
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
from tensorflow import keras
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

#建立VGG13卷积网络，卷积层和全连接层分开

#卷积层
conv_layers = [
    #unit1
    layers.Conv2D(64, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same'),
    
    #unit2
    layers.Conv2D(128, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same'),
    
    #unit3
    layers.Conv2D(256, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same'),

    #unit4
    layers.Conv2D(512, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same'),
    
    #unit5
    layers.Conv2D(512, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3,3], padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same')]

#全连接层
dense_layers = [
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(10, activation=None)]


def preprocess(x,y):
    x = 2* (tf.cast(x, dtype=tf.float32) / 255.) - 1 
    y = tf.cast(y,dtype=tf.int32)
    
    return x,y

#读取数据
(x,y), (x_test, y_test) = datasets.cifar10.load_data()
# [50k,1,100] ==> [50k,100]
#print(y.shape,y_test.shape)
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)

print(x.shape, y.shape, x_test.shape, y_test.shape)

#创建数据集
train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.map(preprocess).shuffle(1000).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(128)

sample = next(iter(train_db))
print('sample:', sample[0].shape,sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[1]))



def main():
    #[b,32,32,3] ==> [b,1,1,512]
    conv_net = Sequential(conv_layers)
    fc_net = Sequential(dense_layers)
    
    conv_net.build(input_shape=[None, 32, 32, 3])
    fc_net.build(input_shape=[None, 512])
    
    optimizer = optimizers.Adam(lr=1e-2)
    
    #将卷积层和全连接层的网络参数拼接到一起
    varibles = conv_net.trainable_variables + fc_net.trainable_variables
    
    for epoch in range(50):
        
        for step, (x,y) in enumerate(train_db):
            
            with tf.GradientTape() as tape:
                out = conv_net(x)
                
                out = tf.reshape(out, [-1,512])
                logits = fc_net(out)
                
                y_onehot = tf.one_hot(y, depth=10)
                
                loss = tf.losses.categorical_crossentropy(y_onehot, logits,from_logits=True)
                
                loss = tf.reduce_mean(loss)
                
            grads = tape.gradient(loss, varibles)
            
            optimizer.apply_gradients(zip(grads, varibles))
            
            if step % 100 == 0:
                print(epoch, step, 'Loss:', float(loss))
        
        #每迭代10次验证：
        total_num = 0
        total_correct = 0
        if epoch % 10 ==0:
            
            for (x_val,y_val) in test_db:
                
                out = conv_net(x_val)
                out = tf.reshape(out, [-1,512])
                logits = fc_net(out)
                
                prob = tf.nn.softmax(logits, axis=1)
                pred = tf.argmax(prob,axis=1)
                #得到的数据类型是int64
                pred = tf.cast(pred,dtype=tf.int32)
                
                correct = tf.cast(tf.equal(pred, y_val), dtype=tf.int32)
                
                correct = tf.reduce_sum(correct)
                
                total_num += x_val.shape[0]
                total_correct += int(correct)
                
            acc = total_correct / total_num
            print(epoch, 'acc:',acc)
                

if __name__ == '__main__':
    main()
    