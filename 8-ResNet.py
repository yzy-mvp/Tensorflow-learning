# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:52:24 2022

@author: swaggy.p
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential

class BasicBlock(layers.Layer):
    #残差模块
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        #第一个卷积模块
        self.conv1 = layers.Conv2D(filter_num, (3,3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        
        #第二个卷积模块
        self.conv2 = layers.Conv2D(filter_num, (3,3), strides=stride, padding='same')
        self.bn2 = layers.BatchNormalization()
        
        if stride != 1:
            self.domnsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1,1), strides=stride))
            
        else:
            #输入输出shape相匹配，之间短接
            self.downsample = lambda x: x
            
    def call(self, inputs, training = None):
        #[b, h, w, c], 通过第一个卷积单元
        out = self.relu(self.bn1(self.conv1(inputs)))
        #第二个卷积单元
        out = self.bn2(self.conv2(out))
        
        #通过identity模块
        identity = self.downsample(inputs)
        #2条路径输出直接相加
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        
        return output
    
class ResNet(keras.Model):
    #通用的ResNet 类
    def __init__(self, layer_dims, num_classes=10): #[2,2,2,2]
        super(ResNet, self).__init__()
        
        #根网络, 预处理
        self.stem = Sequential([layers.Conv2D(64, (3,3), strides=(1,1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2,2), strides=(1,1), padding='same')])
        
        #堆叠4个Block, 每个block包含了2个BasicBlock,
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        
        #通过Pooling将高宽降为1x1
        self.avgpool = layers.GlobalAveragePooling2D()
        
        self.fc = layers.Dense(num_classes)
        
    def call(self, inputs, training=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        
        x =self.avgpool(x)
        x = self.fc(x)
        
        return x
    
    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        #block的第一个BasicBlock的步长为1，实现下采样
        res_blocks.add(BasicBlock(filter_num,stride))
        
        for i in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
            
        return res_blocks
    

def resnet18():
    return ResNet([2,2,2,2])

def resnet34():
    return ResNet([3,4,6,3])

        