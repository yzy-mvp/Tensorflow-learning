# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 23:29:12 2021

@author: swaggy.p
"""
import tensorflow  as tf
import timeit
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###########################cpu与gpu运行时间对比############################

with tf.device('/cpu:0'):
    cpu_a = tf.random.normal([100,200])
    cpu_b = tf.random.normal([200,100])
    print(cpu_a.device, cpu_b.device)
    
with tf.device('/gpu:0'):
    gpu_a = tf.random.normal([100,400])
    gpu_b = tf.random.normal([400,400])
    print(gpu_a.device, gpu_b.device)
    
    
def cpu_run():
    with tf.device('/cpu:0'):
        c = tf.matmul(cpu_a, cpu_b)
        
    return c

def gpu_run():
    with tf.device('/gpu:0'):
        c = tf.matmul(gpu_a, gpu_b)    
    return c


cpu_time = timeit.timeit(cpu_run,number=30)
gpu_time = timeit.timeit(gpu_run, number=30)
print('warm time :',cpu_time, gpu_time)

cpu_time = timeit.timeit(cpu_run,number=30)
gpu_time = timeit.timeit(gpu_run, number=30)
print('run time :',cpu_time, gpu_time)


#tensorflow自动求导
x = tf.constant(1.)
a = tf.constant(2.)
b = tf.constant(3.)
c = tf.constant(4.)

with tf.GradientTape() as tape:
    tape.watch([a,b,c])
    y = a**2 * x + b * x + c
    
[dy_da,dy_db,dy_dc] = tape.gradient(y, [a,b,c])
print(dy_da,dy_db,dy_dc)





    
    