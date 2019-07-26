from keras import layers
from keras import backend as K
from keras.utils import conv_utils
import tensorflow as tf
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util,getGroupSize
from keras import initializers


class GroupConv(layers.Layer): # a scaled layer
    def __init__(self, num_groups_out,kernel_size,h_input, h_output,name='',use_bias=False,bias_initializer='zero',
                 kernel_initializer='he_normal',padding='same',strides=[1, 1, 1, 1],**kwargs):
        super(GroupConv, self).__init__(**kwargs)
        if hasattr(strides,'len') and not len(strides)==4:
            if len(strides)==2:
                strides = [1,strides[0],strides[1],1]
            elif len(strides)==1:
                strides = [1,strides,strides,1]
        elif len(strides)==4:
            pass
        else:
            strides = [1,strides,strides,1]
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.paddingtf=padding.upper()
        self.padding=padding.lower()
        self.stridestf = strides
        self.strides = strides[1:3]
        self.name= name
        self.dilation_rate=[1,1]
        self.data_format='channels_last'
        # for get_config:
        self.num_groups = num_groups_out
        self.h_output  = h_output
        self.h_input = h_input
        if type(kernel_size)==int:
            self.ksize = kernel_size
        else:
            self.ksize = kernel_size[0]
            assert len(kernel_size)==2,kernel_size
        self.kernel_size = [kernel_size,kernel_size]
        #self.kernel_size = ksize
        #self.ksize = kernel_size
    
    def get_config(self):
        config = {
            'kernel_size': self.kernel_size[0],
            'padding': self.padding,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer':initializers.serialize(self.bias_initializer),
            'use_bias':self.use_bias,
            'h_input':self.h_input,
            'h_output':self.h_output,
            'num_groups_out':self.num_groups,
            'name':self.name,
            'strides':self.stridestf
            }
        base_config = super(GroupConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    
    def build(self, input_shape):

        in_channels = input_shape[-1]//getGroupSize(self.h_input)
        self.in_channels = in_channels
        self.gconv_indices, self.gconv_shape_info, self.w_shape = gconv2d_util(
                h_input=self.h_input, h_output=self.h_output, in_channels=in_channels, out_channels=self.num_groups, ksize=self.ksize)

        self.output_dim = input_shape[1]
        self.W = self.add_weight(shape=self.w_shape,initializer=self.kernel_initializer, trainable=True,name=self.name)
        if self.use_bias:
            self.b = self.add_weight(name='bias',shape=self.gconv_shape_info[0:1],trainable=True, initializer=self.bias_initializer)
        #super(GroupConv, self).build(input_shape)  # Be sure to call this somewhere!
        self.built = True
        
    def call(self, x):
        act = gconv2d(input=x, filter=self.W, strides=self.stridestf, padding=self.paddingtf,
                        gconv_indices=self.gconv_indices, gconv_shape_info=self.gconv_shape_info)
        if self.use_bias:
            bias_replicated = tf.tile(self.b,[1,self.gconv_shape_info[1]])
            bias_reshaped = tf.reshape(bias_replicated,[1,1,1,self.num_groups*self.gconv_shape_info[1]])
            #act_reshaped = tf.reshape()
            #act = tf.nn.bias_add(act,bias_replicated)
            act = act + bias_reshaped
        return act
    
    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)   
        #print((input_shape[0],) + tuple(new_space))
        return (input_shape[0],) + tuple(new_space) + tuple([self.gconv_shape_info[0]*self.gconv_shape_info[1]])


class ConvG2Z(layers.Layer):
    def __init__(self, filters,kernel_size,groupType,use_bias=False,bias_initializer='zero',
                 kernel_initializer='he_normal',padding='same',strides=[1, 1, 1, 1],name='',**kwargs):
        super(ConvG2Z, self).__init__(**kwargs)
        if hasattr(strides,'len') and not len(strides)==4:
            if len(strides)==2:
                strides = [1,strides[0],strides[1],1]
            elif len(strides)==1:
                strides = [1,strides,strides,1]
        elif len(strides)==4:
            pass
        else:
            strides = [1,strides,strides,1]
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.paddingtf=padding.upper()
        self.padding=padding.lower()
        self.stridestf = strides
        self.strides = strides[1:3]
        self.name= name
        self.dilation_rate=[1,1]
        self.data_format='channels_last'
        # for get_config:
        self.filters = filters
        self.groupType  = groupType
        if type(kernel_size)==int:
            self.ksize = kernel_size
        else:
            self.ksize = kernel_size[0]
            assert len(kernel_size)==2,kernel_size
        self.kernel_size = [kernel_size,kernel_size]
        #self.kernel_size = ksize
        #self.ksize = kernel_size

    def get_config(self):
        config = {
            'kernel_size': self.kernel_size[0],
            'padding': self.padding,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer':initializers.serialize(self.bias_initializer),
            'use_bias':self.use_bias,
            'groupType':self.groupType,
            'filters':self.filters,
            'name':self.name,
            'strides':self.stridestf
            }
        base_config = super(ConvG2Z, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        group_size = getGroupSize(self.groupType)
        self.group_size = group_size
        in_channels = input_shape[-1]//group_size
        self.in_channels = in_channels
        self.output_dim = input_shape[1]
        print('in',input_shape)
        w_shape = [self.kernel_size[0],self.kernel_size[1],1,self.filters]
        self.W = self.add_weight(shape=w_shape,initializer=self.kernel_initializer, trainable=True,name=self.name)
        #self.pW = self.add_weight(shape=[1,1,in_channels*self.filters*self.group_size,in_channels*self.filters*self.group_size],initializer=initializers.ones(),trainable=False,name=self.name+'pointwise')
        if self.use_bias:
            self.b = self.add_weight(name='bias',shape=self.filters,trainable=True, initializer=self.bias_initializer)
        #super(GroupConv, self).build(input_shape)  # Be sure to call this somewhere!
        self.built = True

    def call(self, x,*kwargs):
        kernel = K.tile(self.W,[1,1,self.group_size*self.in_channels,1])
        print(x)
        print(kernel.shape,K.shape(kernel))
        act=K.depthwise_conv2d(x,depthwise_kernel=kernel,padding=self.padding,
                               data_format=self.data_format)
        #act = K.conv2d(x,kernel,self.strides,padding=self.padding,data_format=self.data_format,dilation_rate=self.dilation_rate)
        if self.use_bias:
            bias_replicated = tf.tile(self.b,[1,self.group_size])
            bias_reshaped = tf.reshape(bias_replicated,[1,1,1,self.filters])
            #act_reshaped = tf.reshape()
            #act = tf.nn.bias_add(act,bias_replicated)
            act = act + bias_reshaped
        #act = K.tile(act,[1,1,1,self.group_size])
        return act

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []

        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        #print((input_shape[0],) + tuple(new_space))
        return (input_shape[0],) + tuple(new_space) + tuple([input_shape[-1]])

def cosetPool(groupType,name):
    if groupType=='C4':
        group_size = 4
    elif groupType == 'D4':
        group_size = 8
    elif groupType == 'Z2':
        group_size = 2
    name = name if name else None
    def call(input):
        fms = K.int_shape(input)
        inp = layers.Reshape([fms[1],fms[2],fms[3]//group_size,group_size])(input)
        #name = name if name else None
        pooled = layers.Lambda(lambda x: tf.reduce_max(x,axis=-1),name=name)(inp)
        #out = layers.Reshape([fms[1],fms[2],fms[3]])(bn)
        return pooled
    return call

def groupedNormFilters(groupType,name=''):
    """
    difference to groupedNormPose: different BatchNorm axis
    :param groupType:
    :return:
    """
    if groupType=='C4':
        group_size = 4
    elif groupType == 'D4':
        group_size = 8
    elif groupType == 'Z2':
        group_size = 2
    def call(input):
        fms = K.int_shape(input)
        inp = layers.Reshape([fms[1],fms[2],fms[3]//group_size,group_size])(input)
        if name:
            bn = layers.BatchNormalization(axis=-2,scale=True,center=True)(inp)
            out = layers.Reshape([fms[1],fms[2],fms[3]],name=name)(bn)
        else:
            bn = layers.BatchNormalization(axis=-2,scale=True,center=True)(inp)
            out = layers.Reshape([fms[1],fms[2],fms[3]])(bn)
        return out
    return call
