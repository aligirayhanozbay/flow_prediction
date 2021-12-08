import numpy as np
import tensorflow as tf

from .ShallowDecoder import _get_original_shallow_decoder_layers
from .SD_UNet import convert_to_keras_tensors
from .FNO import FNO

class SD_FNO(tf.keras.models.Model):
    def __init__(self, input_units, grid_shape, fno_channels, fno_modes, out_channels = 1, sd_config = None, fno_config = None, time_and_space = False):

        if sd_config is None:
            sd_config = {}
        if time_and_space:
            sd_final_units = int(np.prod(grid_shape[1:]))
            inpshape = [grid_shape[0], input_units]
        else:
            sd_final_units = int(np.prod(grid_shape))
            inpshape = [input_units]
        shallow_decoder_layers = _get_original_shallow_decoder_layers(inpshape, sd_final_units, **sd_config)
        shallow_decoder_keras_tensors = convert_to_keras_tensors(shallow_decoder_layers)

        fno_input_shape = [1,*grid_shape] if tf.keras.backend.image_data_format() == 'channels_first' else [*grid_shape,1]
        if time_and_space:
            d = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
            expandeddim_for_channels = tf.expand_dims(shallow_decoder_keras_tensors[-1],d)
            fno_input_reshape_layer = tf.keras.layers.Reshape(fno_input_shape)(expandeddim_for_channels)
        else:
            fno_input_reshape_layer = tf.keras.layers.Reshape(fno_input_shape)(shallow_decoder_keras_tensors[-1])
            

        if fno_config is None:
            fno_config = {}
        fno_final_layer = FNO(fno_input_reshape_layer, out_channels, fno_channels, fno_modes, return_layer=True, **fno_config)

        if time_and_space:
            if tf.keras.backend.image_data_format() == 'channels_first':
                transpose_indices = [0,2,1] + list(range(3,3+len(grid_shape)-1))
                fno_final_layer = tf.transpose(fno_final_layer, transpose_indices)

        super().__init__(shallow_decoder_keras_tensors[0], fno_final_layer)

if __name__ == '__main__':
    ns = 16
    gs = [64,256]
    hc = 16
    modes = [16,16]
    no = 1
    bsize = 20
    inp = tf.random.uniform((bsize,ns))
    mod = SD_FNO(16,gs,hc,modes,out_channels=no)
    mod.summary()
    out = mod(inp)
    print(out.shape)
        

