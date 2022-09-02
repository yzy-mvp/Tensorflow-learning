# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:19:10 2022

@author: swaggy.p
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os

class Generator(keras.Model):
    #生成器网络
    def __init__(self):
        super(Generator, self).__init__()
        
        self.conv1 = layers.Conv2DTranspose(512, 4,1, 'valid',use_bias=False)
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2DTranspose(256, 4,2, 'same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        self.conv3 = layers.Conv2DTranspose(128, 4,2, 'same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        
        self.conv4 = layers.Conv2DTranspose(64, 4,2, 'same', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        
        self.conv5 = layers.Conv2DTranspose(3, 4,2, use_bias=False)
        
        
    def call(self, inputs, training=None):
        x = inputs #[z,100]
        #[b,1,1,100]
        x = tf.reshape(x, (x.shape[0], 1, 1, x.shape[1]))
        x = tf.nn.relu(x)
        
        #[b,1,1,100] => [b,4,4,512]
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        # =>[b,8,8,256]
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
        # =>[b,16,16,128]
        x = tf.nn.relu(self.bn3(self.conv3(x), training=training))
        # =>[b,32,32,64]
        x = tf.nn.relu(self.bn4(self.conv4(x), training=training))
        # =>[b,64,64,3] 
        x = tf.tanh(self.conv5(x))
        
        return x
    
class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
            
        self.conv1 = layers.Conv2D(64, 4,2,'valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(128, 4,2,'valid', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        self.conv3 = layers.Conv2D(256, 4,2,'valid', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        
        self.conv4 = layers.Conv2D(512, 3,1,'valid', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        
        self.conv5 = layers.Conv2D(1024, 3,1,'valid', use_bias=False)
        self.bn5 = layers.BatchNormalization()
        
        self.pool = layers.GlobalAveragePooling2D()
        
        self.flatten = layers.Flatten()
        
        self.fc = layers.Dense(1)
        
    def call(self, inputs, training = None):
        x = tf.nn.leaky_relu(self.bn1(self.conv1(inputs), training=training))
        
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        
        x = tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training))
        
        x = tf.nn.leaky_relu(self.bn5(self.conv5(x), training=training))
        
        x = self.pool(x)
        
        x = self.flatten(x)
        logits = self.fc(x)
        
        return logits

def generate_plot_image(gen_model,test_noise,number):
    pre_images = gen_model(test_noise,training=False)
    plt.figure(figsize=(5,5)) 
    for i in range(pre_images.shape[0]):
        plt.subplot(5,5,i+1)
        plt.imshow((pre_images[i,:,:,0] +1)/2,cmap='gray')
        plt.axis('off')
    plt.savefig(r'D:\jupyter file\GAN\Plots\gen_%d.png'%number)    
    


def celoss_ones(d_real_logits):
    #计算属于与标签1的交叉熵
    y = tf.ones_like(d_real_logits)
    loss = keras.losses.binary_crossentropy(y, d_real_logits, from_logits=True)
    return tf.reduce_mean(loss)

def celoss_zeros(d_fake_logits):
    #计算属于与标签0的交叉熵 
    y = tf.zeros_like(d_fake_logits)
    loss = keras.losses.binary_crossentropy(y, d_fake_logits, from_logits=True)
    return tf.reduce_mean(loss)


def d_loss_fn(generator, discriminator, batch_z, batch_x, training):
    
    fake_image = generator(batch_z, training)
    
    d_fake_logits = discriminator(fake_image, training)
    d_real_logits = discriminator(batch_x, training)
    
    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)
    
    loss = d_loss_fake + d_loss_real
    
    return loss
    
    
def g_loss_fn(generator, discriminator, batch_z, training):
    fake_image = generator(batch_z)
    d_fake_logits = discriminator(fake_image, training)
    
    loss = celoss_ones(d_fake_logits)
    
    return loss


def main():
    tf.random.seed(3333)
    np.random.seed(3333)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
    
    z_dim =100
    epochs = 3000
    batch_size = 64
    learning_rate = 0.0002
    training = True
    
    (train_images, train_labels), (_,_) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0],28,28,1).astype('float32')
    train_images = (train_images - 127.5)/127.5
    
    datasets = train_images
    datasets = tf.data.Dataset.from_tensor_slices(datasets,)
    datasets = datasets.shuffle(100000).batch(batch_size).repeat(100)
    db_iter = datasets.as_numpy_iterator()
    
    
    generator = Generator()
    generator.build(input_shape=(None,z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None,28,28,1))
    #创建优化器
    g_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    
    d_losses = []
    g_losses = []
    num_epochs = []
    for epoch in range(epochs):
        
        #训练鉴别器
        for _ in range(1):
            #采样隐藏向量
            batch_z = tf.random.normal([batch_size, z_dim])
            #采样真实数据
            batch_x = next(db_iter)
            
            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, training)
                
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
            
        #采样隐藏向量
        batch_z = tf.random.normal([batch_size, z_dim])
        #采样真实数据
        batch_x = next(db_iter)
        
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, training)
            
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        
        if epoch % 100 == 0:
            print(epoch, 'd-loss:',float(d_loss), 'g-loss:',float(g_loss))
            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))
            num_epochs.append(epoch)
                    
            #生成图片
            num_exp_to_generate = 25
            seed = tf.random.normal([num_exp_to_generate,z_dim])#测试种子
            #pre_images = generator(seed,training=False)
            #pre_images = tf.reshape(pre_images,[25,28,28])
            generate_plot_image(generator,seed,epoch)
            
            



if __name__ =='__main__':
    main()
            
        
        
        
        
        
        
