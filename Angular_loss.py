#!/usr/bin/env python
# coding: utf-8

# In[13]:


import keras
from keras.engine.topology import Layer
from keras import backend as K
from keras.initializers import Constant
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects
from keras.models import Model
import math


# In[14]:


class Dense_with_Arcface_loss(Layer): #Perfection ! 

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Dense_with_Arcface_loss, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Dense_with_Arcface_loss, self).build(input_shape)
        #self.kernel = K.l2_normalize(self.kernel,0)

    def call(self, inputs):
        inputs = tf.nn.l2_normalize(inputs, dim=1)  # input_l2norm
        self.kernel = tf.nn.l2_normalize(self.kernel, dim=0)   # W_l2norm

        cosine = K.dot(inputs, self.kernel)  # cos = input_l2norm * W_l2norm
        return cosine

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(Dense_with_Arcface_loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# In[15]:


def ArcFace_Loss(y_true, y_pred,m=0.35,scale=50.):
        #Parameters  
        num_classes=y_true.shape[1]
        cos_m = math.cos(m) 
        sin_m = math.sin(m) 
        threshold = math.cos(math.pi - m)
        mm = math.sin(math.pi - m) * m
        
        cos_t = y_pred
        
        cos_t2 = K.tf.square(cos_t, name='cos_2')
        sin_t2 = K.tf.subtract(1., cos_t2, name='sin_2')
        sin_t = K.tf.sqrt(sin_t2, name='sin_t')
        cos_mt = scale * tf.subtract(K.tf.multiply(cos_t, cos_m), K.tf.multiply(sin_t, sin_m), name='cos_mt')

        cond_v = cos_t - threshold
        cond = tf.cast(K.tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = scale*(cos_t - mm)
        cos_mt_temp = K.tf.where(cond, cos_mt, keep_val)

        mask = y_true
        
        inv_mask = K.tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = K.tf.multiply(scale, cos_t, name='scalar_cos_t')

        output = K.tf.add(K.tf.multiply(s_cos_t, inv_mask), K.tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')

        cross_ent = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=output)
        loss = tf.reduce_mean(cross_ent)
        
        return loss


# In[16]:


def CosineFace_Loss(y_true, y_pred,m=0.35,scale=30.):
    num_classes=y_true.shape[1]
    cosine = y_pred

    cosine = tf.clip_by_value(cosine, -1, 1, name='cosine_clip') - m * y_true
    
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
        logits=scale * cosine), name='cosine_loss')


# In[17]:


from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Conv2D, BatchNormalization, GlobalMaxPooling2D
from keras.optimizers import Adam

def create_model(input_shape,backbone,lr,num_classes,loss='arcface'):
    
    inp = Input(shape=input_shape)
    
    for layer in backbone.layers:
        layer.trainable = True
        
    feature = backbone(inp)
    feature = GlobalAveragePooling2D()(feature) # 0.294
    #feature = GlobalMaxPooling2D()(feature)
    feature = BatchNormalization()(feature)
    

        
    Arcface_softmax_logits = Dense_with_Arcface_loss(num_classes)
    out = Arcface_softmax_logits(feature)
    model = Model(inputs=inp, outputs=out)
    if loss=='cosineface':
        model.compile(loss=CosineFace_Loss, optimizer='Adam', metrics=['acc'])
    if loss=='arcface':
        model.compile(loss=ArcFace_Loss, optimizer='Adam', metrics=['acc'])
    
    return model


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




