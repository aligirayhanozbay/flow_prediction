import copy
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
import tensorflow as tf
import numpy as np

from .ShallowDecoder import _get_original_shallow_decoder_layers
from .SD_UNet import convert_to_keras_tensors
from .TemporalUNet import TemporalUNet

class SD_TemporalUNet(tf.keras.models.Model):
    def __init__(self, input_units, grid_shape, out_channels = 1, sd_config = None, temporalunet_config = None, nt = None):
        
        if sd_config is None:
            sd_config = {}
        sd_final_units = int(np.prod(grid_shape))
        shallow_decoder_layers = _get_original_shallow_decoder_layers([nt,input_units], sd_final_units, **sd_config)
        shallow_decoder_keras_tensors = convert_to_keras_tensors(shallow_decoder_layers)
        
        unet_input_shape = [nt,1,*grid_shape] if tf.keras.backend.image_data_format() == 'channels_first' else [nt,*grid_shape,1]
        unet_input_reshape_layer = tf.keras.layers.Reshape(unet_input_shape)(shallow_decoder_keras_tensors[-1])

        if temporalunet_config is None:
            temporalunet_config = {}
        unet_final_layer = TemporalUNet(input_layer = unet_input_reshape_layer, nt = None,  nx = None, ny = None, in_channels=None, out_channels=out_channels, padding='same', return_layer = True, **temporalunet_config)

        super().__init__(shallow_decoder_keras_tensors[0], unet_final_layer)


if __name__=='__main__':
    
    #tf.keras.backend.set_image_data_format('channels_first')
    tf.keras.backend.set_image_data_format('channels_last')


    nf = 16
    nx = 64
    ny = 256
    nt = 10
    nc = 2
    bsize = 5
    inp = tf.random.uniform((bsize,nt,nf))

    #model = RNN_TemporalUNet(nf, [nx, ny], out_channels = nc, nt = nt)
    model = SD_TemporalUNet(nf, [nx, ny], out_channels = nc, nt = nt)
    model.summary()
