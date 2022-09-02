# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:06:28 2022

@author: swaggy.p
"""
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, Sequential, datasets

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

batchsz = 128
total_words = 10000   #词汇表大小
max_review_len = 80   #句子的最大长度，大于的部分将截断，小于的部分将补充
enbedding_len = 100  #词向量的特征长度

#加载数据集, 一个数字表示一个单词
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=total_words)
print(x_train.shape, len(x_train[0]), y_train.shape)
print(x_test.shape, len(x_test[0]), y_test.shape)

#数字编码表
'''
word_index = datasets.imdb.get_word_index()
for k,v in word_index.items():
    print(k,v)
'''

#x_train: [b,80]
#x_test: [b,80]
#截断和填充句子
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)
#构建数据集， 打散、批量、丢掉最后一次不够一个batch的数据
db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)

db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)

print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)


class MyRnn(keras.Model):
    #cell方式构建多层网络
    def __init__(self,units):
        super(MyRnn, self).__init__()
        #初始化cell的状态向量
        self.state0 = [tf.zeros([batchsz,units])]
        self.state1 = [tf.zeros([batchsz,units])]
        
        #词向量编码： [b,80] =>[b,80,100]
        self.enbedding = layers.Embedding(total_words, enbedding_len, input_length=max_review_len)
        
        #构建两个cell
        self.rnn_cell0 = layers.SimpleRNNCell(units,dropout=0.5)
        self.rnn_cell1 = layers.SimpleRNNCell(units,dropout=0.5)
        
        #构建分类网络 
        #[b, 80, 100] => [b,64] => [b,1]
        self.outlayer = Sequential([layers.Dense(units),
                                    layers.Dropout(0.5),
                                    layers.ReLU(),
                                    layers.Dense(1)])
        
        
    def call(self, inputs, training=None):
        #inputs: [b, 80]
        x = self.enbedding(inputs)
        # x: [b, 80,100]
        
        state0 = self.state0
        state1 = self.state1
        
        for word in tf.unstack(x,axis=1):
            #把axis=1维度的数展开
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell1(out0,state1, training)
            
        x = self.outlayer(out1, training)
        prob = tf.sigmoid(x)
        
        return prob
    
    
def main():
    
    units = 64
    epochs = 50
    
    model = MyRnn(units)
    model.compile(optimizer=optimizers.RMSprop(0.001),
                  loss=losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    
    #训练和验证
    model.fit(db_train, epochs=epochs, validation_data=db_test)
    model.evaluate(db_test)
    

if __name__ == '__main__':
    main()
    
        
        
        