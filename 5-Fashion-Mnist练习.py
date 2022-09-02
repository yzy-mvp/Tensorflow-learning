# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 13:43:19 2022

@author: swaggy.p
"""
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics


def process(x,y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    
    return x,y
    

(x, y),(x_test, y_test) = datasets.fashion_mnist.load_data()

batchsz = 128
db = tf.data.Dataset.from_tensor_slices((x,y))
db = db.map(process).shuffle(10000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(process).batch(batchsz)

db_iter = iter(db)

sample = next(db_iter)
print('Batch:',sample[0].shape,sample[1].shape)


model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)])

model.build(input_shape=[None,28*28])
#model.summary()

optimize = optimizers.Adam(learning_rate=1e-3)




def main():
    
    for epoch in range(30):
        
        for step, (x,y) in enumerate(db):
            x = tf.reshape(x,[-1,28*28])
            
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=10)
                
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits,from_logits=True))
                
            grads = tape.gradient(loss_ce, model.trainable_variables)
            
            optimize.apply_gradients(zip(grads,model.trainable_variables))
            
            
            if step% 100 == 0:
                print(epoch, step,'Loss:',float(loss_ce))  
        
        #test
        total_correct = 0
        total_number = 0
        
        for x,y in db_test:
            x = tf.reshape(x,[-1,28*28])
            
            logits = model(x)
            
            prob = tf.nn.softmax(logits, axis=1)
            
            pred = tf.argmax(prob,axis=1)
            pred = tf.cast(pred,dtype=tf.int32)
            correct = tf.equal(pred, y)
            
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
            
            total_correct += correct
            
            total_number += x.shape[0]
            
        acc = total_correct / total_number
        print(epoch, 'Test acc:',acc)
        
        
            
            
            
            
                
                
                


if __name__ == '__main__':
    main()