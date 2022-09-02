# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 11:24:27 2022

@author: swaggy.p
"""
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow import keras
import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x,y):
    x = (tf.cast(x, dtype=tf.float32)-127.5) /127.5
    y = tf.cast(y,dtype=tf.int32)
    
    return x,y


batchsz =128
#加载数据集
(x,y), (x_val,y_val) = datasets.cifar10.load_data()

y = tf.squeeze(y)
y = tf.one_hot(y, depth=10)
y_val = tf.squeeze(y_val)
y_val = tf.one_hot(y_val, depth=10)
print('datasets:',x.shape,y.shape, x.min(), x.max())

train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.map(preprocess).shuffle(1000).batch(batchsz)

test_db = tf.data.Dataset.from_tensor_slices((x_val,y_val))
test_db = test_db.map(preprocess).batch(batchsz)

sample = next(iter(train_db))
print('Batch:',sample[0].shape,sample[1].shape)



#自定义Dense
class MyDense(layers.Layer):
    
    def __init__(self, input_dim,output_dim):
        super(MyDense,self).__init__()
        
        self.kernel = self.add_weight('w', [input_dim, output_dim])
        #self.bias = self.add_variable('b', [output_dim])
        
    def call(self, inputs, training = None):
        
        x = inputs @ self.kernel 
        
        return x
    
    
#自定义Model
class MyNetwork(keras.Model):
    
    def __init__(self):
        super(MyNetwork,self).__init__()
        
        self.fc1 = MyDense(32*32*3,512)
        self.fc2 = MyDense(512, 256)
        self.fc3 = MyDense(256, 128)
        self.fc4 = MyDense(128, 100)
        self.fc5 = MyDense(100, 10)
        
    def call(self, inputs,training=None):
        '''
        Parameters
        ----------
        inputs : [b,32*32*3]
        
            DESCRIPTION.
        training : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        [b,10]

        '''
        x = tf.reshape(inputs, shape=[-1,32*32*3])
        #x = keras.layers.Flatten(inputs)
        x = tf.nn.relu(self.fc1(x))
        x = tf.nn.relu(self.fc2(x))
        x = tf.nn.relu(self.fc3(x))
        x = tf.nn.relu(self.fc4(x))
        x = self.fc5(x)
        
        return x
    
network = MyNetwork()   

network.compile(optimizer=optimizers.Adam(lr=1e-1),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

network.fit(train_db,epochs=5,validation_data=test_db, validation_freq=1)

network.evaluate(test_db)
#network.save_weights('./weights.ckpt')
#del network

print('saved to ckp/weights.ckpt')



        
        
        
        
        
    






