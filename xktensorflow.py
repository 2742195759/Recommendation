import tensorflow as tf
import numpy as np
# the python file is a collection of 
# tensorflow function based on tensorflow
# xktf_XXX means the function return a tf_tensor and should be s.run() 
# xknp_XXX means it return the real val , this means it may be written
#   by tensorflow operate

def multi_hots(indices , depth , device='') : 
    if device :
        with tf.device(device) :
            return tf.reduce_sum(tf.one_hot(indices , depth) , axis=-2)
    
    return tf.reduce_sum(tf.one_hot(indices , depth) , axis=-2)

