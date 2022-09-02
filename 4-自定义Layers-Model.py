# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 16:36:59 2022

@author: swaggy.p
"""
import tensorflow  as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import tensorflow.keras as  keras

def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y


batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets:', x.shape, y.shape, x.min(), x.max())



db = tf.data.Dataset.from_tensor_slices((x,y))
db = db.map(preprocess).shuffle(60000).batch(batchsz)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz) 

sample = next(iter(db))
print(sample[0].shape, sample[1].shape)



#自定义layder: 继承keras.layers.Layer,有__init__()和call()方法
class MyDense(keras.layers.Layer):
    
    def __init__(self, input_shape, out_shape):
        super(MyDense, self).__init__()   #子类的对象继承父类的对象，并调用父类的__init__()方法初始化子类属性
        
        self.kernel = self.add_weight('w',[input_shape,out_shape])
        self.bias = self.add_weight('b',[out_shape])
        
    def call(self,inputs,training=None):
    
        out = inputs @ self.kernel + self.bias
        
        return out
    
#自定义网络Model: 继承kears.Model, 具有__init__()和call()方法
class MyModel(keras.Model):
    
    def __init__(self):
        super(MyModel,self).__init__()
        
        self.fc1 = MyDense(28*28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)
        
    def call(self,inputs, training=None):
        x = tf.nn.relu(self.fc1(inputs))
        x = tf.nn.relu(self.fc2(x))
        x = tf.nn.relu(self.fc3(x))
        x = tf.nn.relu(self.fc4(x))
        x = tf.nn.relu(self.fc5(x))
        
        return x


network = MyModel()

network.compile(optimizer=optimizers.Adam(learning_rate=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

network.fit(db,epochs=5,validation_data=ds_val,validation_freq=2)

#整个模型的保存与加载
#network.save('model.h5')
#del network
#network = keras.models.load_model('model.h5')