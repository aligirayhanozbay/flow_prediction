import numpy as np
import tensorflow as tf
from typing import Optional, Union, Callable, List

from .UNet import _get_padding_sizes, _get_kernel_initializer, _get_filter_count, slrelu

class TemporalConvBlock(tf.keras.layers.Layer):
    normalization_map = {'batchnorm': tf.keras.layers.BatchNormalization, 'layernorm': tf.keras.layers.LayerNormalization, None: lambda *args,**kwargs: lambda x: x}
    def __init__(self, layer_idx, filters_root, kernel_size, dropout_rate, recurrent_dropout_rate, padding, activation, recurrent_activation, normalization = None, padding_mode = None, **kwargs):
        super().__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.dropout_rate=dropout_rate
        self.recurrent_dropout_rate=recurrent_dropout_rate
        self.activation=activation
        self.recurrent_activation=recurrent_activation
        self.normalization=normalization

        assert (padding.lower() in ['same', 'valid'])
        self.padding=padding.lower()
        assert (padding_mode is None) or (padding_mode.lower() in ['constant', 'reflect', 'symmetric'])
        self.padding_mode = padding_mode.lower() if padding_mode is not None else 'constant'

        self.normalization_1 = self._get_normalization_layer()
        self.conv_1 = self._get_conv_layer()
        self._conv1_paddings_size = _get_padding_sizes((self.kernel_size,self.kernel_size), include_time_dim=True)

        self.normalization_2 = self._get_normalization_layer()
        self.conv_2 = self._get_conv_layer()
        self._conv2_paddings_size = _get_padding_sizes((self.kernel_size,self.kernel_size), include_time_dim=True)

    def _get_normalization_layer(self):
        return self.normalization_map[self.normalization](axis=2 if tf.keras.backend.image_data_format() == 'channels_first' else -1)
    
    def _get_conv_layer(self):
        filters = _get_filter_count(self.layer_idx, self.filters_root)
        layer = tf.keras.layers.ConvLSTM2D(filters = filters,
                                           kernel_size = (self.kernel_size, self.kernel_size),
                                           kernel_initializer = _get_kernel_initializer(filters, self.kernel_size),
                                           strides=1,
                                           padding="valid",
                                           activation=self.activation,
                                           recurrent_activation=self.recurrent_activation,
                                           return_state=False,
                                           return_sequences=True,
                                           dropout=self.dropout_rate,
                                           recurrent_dropout=self.recurrent_dropout_rate)
        return layer

    def _pad(self, x, paddings):
        return tf.pad(x, paddings, mode=self.padding_mode)

    def call(self, inputs, training=None, **kwargs):
        x = inputs

        x = self.normalization_1(x)
        x = self._pad(x, self._conv1_paddings_size)
        x = self.conv_1(x, training=training)
        
        
        x = self.normalization_2(x)
        x = self._pad(x, self._conv2_paddings_size)
        x = self.conv_2(x,training=training)
         
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    dropout_rate=self.dropout_rate,
                    recurrent_dropout_rate=self.recurrent_dropout_rate,
                    padding=self.padding,
                    padding_mode=self.padding_mode,
                    activation=self.activation,
                    recurrent_activation=self.recurrent_activation,
                    normalization=self.normalization,
                    **super().get_config(),
                    )

class TemporalUpconvBlock(tf.keras.layers.Layer):
    normalization_map = {'batchnorm': tf.keras.layers.BatchNormalization, 'layernorm': tf.keras.layers.LayerNormalization, None: lambda *args,**kwargs: lambda x: x}
    transpose_indices_map = {'channels_first': [0,2,1,3,4], 'channels_last': [0,4,2,3,1]}
    def __init__(self, layer_idx, filters_root, temporal_kernel_size, pool_size, padding, activation, normalization = None, temporal_stride=1, **kwargs):
        super().__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.temporal_kernel_size=temporal_kernel_size
        self.pool_size=pool_size
        self.activation=activation

        assert (padding.lower() in ['same', 'valid'])
        self.padding=padding.lower()

        ksize = (temporal_kernel_size, pool_size, pool_size)
        filters = _get_filter_count(layer_idx + 1, filters_root)
        norm_axis=2 if tf.keras.backend.image_data_format() == 'channels_first' else -1
        self.normalization = self.normalization_map[normalization](axis=norm_axis)
        self.upconv = tf.keras.layers.Conv3DTranspose(filters // 2,
                                            kernel_size=ksize,
                                            activation=activation,
                                            strides=(temporal_stride,pool_size,pool_size),
                                            padding=self.padding)

        self._transpose_indices = self.transpose_indices_map[tf.keras.backend.image_data_format()]
        self._paddings_size = _get_padding_sizes(ksize)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.normalization(x)
        x = tf.transpose(x,self._transpose_indices)
        x = self.upconv(x)
        x = tf.transpose(x,self._transpose_indices)
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    temporal_kernel_size=self.temporal_kernel_size,
                    pool_size=self.pool_size,
                    padding=self.padding,
                    activation=self.activation,
                    padding_mode=self.padding_mode
                    **super().get_config(),
                    )

class TemporalUNetCropConcatBlock(tf.keras.layers.Layer):

    def call(self, x, down_layer, **kwargs):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        if tf.keras.backend.image_data_format() == 'channels_last':
            height_diff = (x1_shape[2] - x2_shape[2]) // 2
            width_diff = (x1_shape[3] - x2_shape[3]) // 2
            down_layer_cropped = down_layer[:,:,
                                        height_diff: (x1_shape[2] - height_diff),
                                        width_diff: (x1_shape[3] - width_diff),
                                            :]
        else:
            height_diff = (x1_shape[3] - x2_shape[3]) // 2
            width_diff = (x1_shape[4] - x2_shape[4]) // 2
            down_layer_cropped = down_layer[:,:,:,
                                        height_diff: (x1_shape[3] - height_diff),
                                        width_diff: (x1_shape[4] - width_diff)]

        x = tf.concat([down_layer_cropped, x], axis=-1 if tf.keras.backend.image_data_format() == 'channels_last' else 2)
        return x

class TemporalUNetPoolingBlock(tf.keras.layers.AveragePooling3D):
    transpose_indices_map = {'channels_first': [0,2,1,3,4], 'channels_last': [0,4,2,3,1]}
    def call(self, x, *args, **kwargs):
        x = tf.transpose(x, self.transpose_indices_map[self.data_format])
        x = super().call(x)
        x = tf.transpose(x, self.transpose_indices_map[self.data_format])
        return x


def TemporalUNet(
        input_layer: Optional[tf.keras.layers.Layer] = None,
        nt: Optional[int] = None,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        in_channels: int = 1,
        out_channels: int = 1,
        layer_depth: int = 5,
        filters_root: int = 64,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout_rate: float = 0.5,
        recurrent_dropout_rate: float = 0.5,
        padding:str="valid",
        padding_mode:str="constant",
        activation:Union[str, Callable]="relu",
        recurrent_activation:Union[str, Callable]="sigmoid",
        final_activation:Union[str,Callable]="linear",
        final_kernel_size: int = None,
        return_layer=False,
        normalization=None):
    """
    Constructs a U-Net model
    :param nx: (Optional) image size on x-axis
    :param ny: (Optional) image size on y-axis
    :param in_channels: number of input channels of the input tensors
    :param out_channels: number of channels of the output
    :param layer_depth: total depth of unet
    :param filters_root: number of filters in top unet layer
    :param kernel_size: size of convolutional layers
    :param pool_size: size of maxplool layers
    :param dropout_rate: rate of dropout
    :param padding: padding to be used in convolutions
    :param activation: activation to be used
    :return: A TF Keras model
    """
    if input_layer is None:
        inpshape = (nt, nx, ny, in_channels) if tf.keras.backend.image_data_format() == 'channels_last' else (nt, in_channels,nx,ny)
        inputs = tf.keras.layers.Input(shape=inpshape, name="inputs")
        x = inputs
    else:
        x = input_layer

    if activation == 'leaky_relu':
        activation = tf.nn.leaky_relu
    elif activation == 'slrelu':
        activation = slrelu

    if final_kernel_size is None:
        final_kernel_size = kernel_size
        
    contracting_layers = {}

    conv_params = dict(filters_root=filters_root,
                       kernel_size=kernel_size,
                       dropout_rate=dropout_rate,
                       recurrent_dropout_rate=recurrent_dropout_rate,
                       padding=padding,
                       padding_mode=padding_mode,
                       activation=activation,
                       recurrent_activation=recurrent_activation,
                       normalization=normalization)

    for layer_idx in range(0, layer_depth - 1):
        x = TemporalConvBlock(layer_idx, **conv_params)(x)
        contracting_layers[layer_idx] = x
        x = TemporalUNetPoolingBlock((1,pool_size, pool_size), padding = 'same', data_format = tf.keras.backend.image_data_format())(x)

    x = TemporalConvBlock(layer_idx + 1, **conv_params)(x)

    for layer_idx in range(layer_idx, -1, -1):
        x = TemporalUpconvBlock(layer_idx,
                        filters_root,
                        kernel_size,
                        pool_size,
                        padding,
                        activation,
                        normalization=normalization)(x)
        x = TemporalUNetCropConcatBlock()(x, contracting_layers[layer_idx])
        x = TemporalConvBlock(layer_idx, **conv_params)(x)

    outputs = tf.keras.layers.ConvLSTM2D(filters=out_channels,
                                kernel_size=final_kernel_size,
                                strides=1,
                                padding=padding,
                                activation=final_activation,
                                return_state=False,
                                return_sequences=True)(x)

    if return_layer:
        return outputs
    else:
        model = tf.keras.Model(inputs, outputs, name="temporalunet")
        return model

if __name__=='__main__':
    
    tf.keras.backend.set_image_data_format('channels_first')
    
    nx = 64
    ny = 256
    nt = 10
    nc = 2
    bsize = 5
    inp = tf.random.uniform((bsize,nt,nc,nx,ny))

    model = TemporalUNet(nt=nt, nx=nx, ny=ny, in_channels=nc, out_channels=1, layer_depth=4, padding='same', filters_root=16)
    model.summary()

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        out = model(inp)
        print(out.shape)
        loss_val = tf.reduce_mean(tf.abs(out-(inp+2)))
    grads = tape.gradient(loss_val, model.trainable_variables)
    assert tf.reduce_all([tf.reduce_all(tf.math.logical_not(tf.math.is_nan(x))) for x in grads])

    # cbl = TemporalConvBlock(1,16,3,0.0,0.0,'same','relu','sigmoid')
    # ubl = TemporalUpconvBlock(1,16,3,2,'same','relu')
    # pbl = TemporalUNetPoolingBlock((1,2,2), padding='valid', data_format='channels_first')
    # ccbl = TemporalUNetCropConcatBlock()

    # print(inp.shape)
    # co = cbl(inp)
    # print(co.shape)
    # cp = pbl(inp)
    # print(cp.shape)
    # cu = ubl(cp)
    # print(cu.shape)
    # cc = ccbl(inp)
    # print(cc.shape)
