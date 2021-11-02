'''
TF 2.0 U-Net implementation based on the following Github repository:
https://github.com/jakeret/unet

@article{akeret2017radio,
  title={Radio frequency interference mitigation using deep convolutional neural networks},
  author={Akeret, Joel and Chang, Chihway and Lucchi, Aurelien and Refregier, Alexandre},
  journal={Astronomy and Computing},
  volume={18},
  pages={35--39},
  year={2017},
  publisher={Elsevier}
}
'''

from typing import Optional, Union, Callable, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import Adam

def _get_filter_count(layer_idx, filters_root):
    return 2 ** layer_idx * filters_root


def _get_kernel_initializer(filters, kernel_size):
    stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
    return TruncatedNormal(stddev=stddev)


class ConvBlock(layers.Layer):
    normalization_map = {'batchnorm': tf.keras.layers.BatchNormalization, 'layernorm': tf.keras.layers.LayerNormalization, None: lambda *args,**kwargs: lambda x: x}
    def __init__(self, layer_idx, filters_root, kernel_size, dropout_rate, padding, activation, normalization = None, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.dropout_rate=dropout_rate
        self.padding=padding
        self.activation=activation

        filters = _get_filter_count(layer_idx, filters_root)
        self.normalization_1 = self.normalization_map[normalization](axis=1 if tf.keras.backend.image_data_format() == 'channels_first' else -1)
        self.conv2d_1 = layers.Conv2D(filters=filters,
                                      kernel_size=(kernel_size, kernel_size),
                                      kernel_initializer=_get_kernel_initializer(filters, kernel_size),
                                      strides=1,
                                      padding=padding)
        self.dropout_1 = layers.Dropout(rate=dropout_rate)
        self.activation_1 = layers.Activation(activation)

        self.normalization_2 = self.normalization_map[normalization](axis=1 if tf.keras.backend.image_data_format() == 'channels_first' else -1)
        self.conv2d_2 = layers.Conv2D(filters=filters,
                                      kernel_size=(kernel_size, kernel_size),
                                      kernel_initializer=_get_kernel_initializer(filters, kernel_size),
                                      strides=1,
                                      padding=padding)
        self.dropout_2 = layers.Dropout(rate=dropout_rate)
        self.activation_2 = layers.Activation(activation)
        

    def call(self, inputs, training=None, **kwargs):
        x = inputs

        x = self.normalization_1(x)
        x = self.conv2d_1(x)
        if training:
            x = self.dropout_1(x)
        x = self.activation_1(x)
        
        x = self.normalization_2(x)
        x = self.conv2d_2(x)
        if training:
            x = self.dropout_2(x)
        x = self.activation_2(x)
        
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    dropout_rate=self.dropout_rate,
                    padding=self.padding,
                    activation=self.activation,
                    **super(ConvBlock, self).get_config(),
                    )


class UpconvBlock(layers.Layer):
    normalization_map = {'batchnorm': tf.keras.layers.BatchNormalization, 'layernorm': tf.keras.layers.LayerNormalization, None: lambda *args,**kwargs: lambda x: x}
    def __init__(self, layer_idx, filters_root, kernel_size, pool_size, padding, activation, normalization = None, **kwargs):
        super(UpconvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.pool_size=pool_size
        self.padding=padding
        self.activation=activation

        filters = _get_filter_count(layer_idx + 1, filters_root)
        self.normalization = self.normalization_map[normalization](axis=1 if tf.keras.backend.image_data_format() == 'channels_first' else -1)
        self.upconv = layers.Conv2DTranspose(filters // 2,
                                             kernel_size=(pool_size, pool_size),
                                             kernel_initializer=_get_kernel_initializer(filters, kernel_size),
                                             strides=pool_size, padding=padding)

        self.activation_1 = layers.Activation(activation)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.normalization(x)
        x = self.upconv(x)
        x = self.activation_1(x)
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    pool_size=self.pool_size,
                    padding=self.padding,
                    activation=self.activation,
                    **super(UpconvBlock, self).get_config(),
                    )

class CropConcatBlock(layers.Layer):

    def call(self, x, down_layer, **kwargs):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        if tf.keras.backend.image_data_format() == 'channels_last':
            height_diff = (x1_shape[1] - x2_shape[1]) // 2
            width_diff = (x1_shape[2] - x2_shape[2]) // 2
            down_layer_cropped = down_layer[:,
                                        height_diff: (x1_shape[1] - height_diff),
                                        width_diff: (x1_shape[2] - width_diff),
                                            :]
        else:
            height_diff = (x1_shape[2] - x2_shape[2]) // 2
            width_diff = (x1_shape[3] - x2_shape[3]) // 2
            down_layer_cropped = down_layer[:,:,
                                        height_diff: (x1_shape[2] - height_diff),
                                        width_diff: (x1_shape[3] - width_diff)]

        x = tf.concat([down_layer_cropped, x], axis=-1 if tf.keras.backend.image_data_format() == 'channels_last' else 1)
        return x

@tf.function(experimental_relax_shapes=True)
def slrelu(x, alpha=0.2):
    #https://stats.stackexchange.com/questions/329776/approximating-leaky-relu-with-a-differentiable-function/329803
    return alpha*x + (1-alpha) * tf.math.log(1+tf.exp(x))
    
def UNet(
        input_layer: Optional[tf.keras.layers.Layer] = None,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        in_channels: int = 1,
        out_channels: int = 1,
        layer_depth: int = 5,
        filters_root: int = 64,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout_rate: int = 0.5,
        padding:str="valid",
        activation:Union[str, Callable]="relu",
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
        inpshape = (nx, ny, in_channels) if tf.keras.backend.image_data_format() == 'channels_last' else (in_channels,nx,ny)
        inputs = Input(shape=inpshape, name="inputs")
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
                       padding=padding,
                       activation=activation,
                       normalization=normalization)

    for layer_idx in range(0, layer_depth - 1):
        x = ConvBlock(layer_idx, **conv_params)(x)
        contracting_layers[layer_idx] = x
        x = layers.AveragePooling2D((pool_size, pool_size), padding = 'same', data_format = tf.keras.backend.image_data_format())(x)

    x = ConvBlock(layer_idx + 1, **conv_params)(x)

    for layer_idx in range(layer_idx, -1, -1):
        x = UpconvBlock(layer_idx,
                        filters_root,
                        kernel_size,
                        pool_size,
                        padding,
                        activation,
                        normalization=normalization)(x)
        x = CropConcatBlock()(x, contracting_layers[layer_idx])
        x = ConvBlock(layer_idx, **conv_params)(x)

    x = layers.Conv2D(filters=out_channels,
                      kernel_size=final_kernel_size,
                      kernel_initializer=_get_kernel_initializer(filters_root, final_kernel_size),
                      strides=1,
                      padding=padding)(x)

    x = layers.Activation(activation)(x)
    outputs = layers.Activation(final_activation, name="outputs")(x)

    if return_layer:
        return outputs
    else:
        model = Model(inputs, outputs, name="unet")
        return model
