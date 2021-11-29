import copy
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
import tensorflow as tf
import numpy as np

from .ShallowDecoder import _get_original_shallow_decoder_layers
from .UNet import slrelu
from .SD_UNet import convert_to_keras_tensors
from .TemporalUNet import TemporalUNet

def _handle_activations(cfg_dict):
    if 'activation' in cfg_dict:
        if cfg_dict['activation'] == 'leaky_relu':
            cfg_dict['activation'] = tf.nn.leaky_relu
        elif cfg_dict['activation'] == 'slrelu':
            cfg_dict['activation'] = slrelu
    if 'recurrent_activation' in cfg_dict:
        if cfg_dict['recurrent_activation'] == 'leaky_relu':
            cfg_dict['recurrent_activation'] = tf.nn.leaky_relu
        elif cfg_dict['recurrent_activation'] == 'slrelu':
            cfg_dict['recurrent_activation'] = slrelu
        

def _RNNStack_layers(input_layer_shape, output_layer_size, hidden_layer_units=None, hidden_layer_config = None, final_layer_config = None, normalization = None, rnn_type = None, nt=None):
    
        
    if rnn_type is None or rnn_type.lower() == 'lstm':
        rnn_type = lambda *args, **kwargs: tf.keras.layers.LSTM(*args,
                                                                return_sequences = True,
                                                                stateful=False,
                                                                return_state=False,
                                                                **kwargs)
        if tf.keras.backend.image_data_format() == 'channels_first':
            import warnings
            warnings.warn('Layers created by this function may fail if tf.keras.backend.image_data_format() is channels_first due to a TF bug. See this issue: https://github.com/tensorflow/tensorflow/issues/48364')
    elif rnn_type.lower() == 'gru':
        rnn_type = lambda *args, **kwargs: tf.keras.layers.GRU(*args,
                                                                return_sequences = True,
                                                                stateful=False,
                                                                return_state=False,
                                                                **kwargs)
    else:
        raise(ValueError(f'rnn_type must be lstm or gru, you provided: {rnn_type}'))

    if hidden_layer_units is None:
        hidden_layer_units = [40,40]
    if hidden_layer_config is None:
        hidden_layer_config = {}
    if final_layer_config is None:
        final_layer_config = {}
    hidden_layer_config = copy.deepcopy(hidden_layer_config)
    final_layer_config = copy.deepcopy(final_layer_config)
    _handle_activations(hidden_layer_config)
    _handle_activations(final_layer_config)

    normalization_map = {'batchnorm': tf.keras.layers.BatchNormalization, 'layernorm': tf.keras.layers.LayerNormalization}

    layers = [tf.keras.layers.Input([nt,input_layer_shape])]
    for units in hidden_layer_units:
        layers.append(
            rnn_type(units, **hidden_layer_config)
        )
        if normalization is not None:
            layers.append(normalization_map[normalization]())
    layers.append(rnn_type(output_layer_size, **final_layer_config))
    return layers

def RNNStack(*layers_args, optimizer = None, loss = None, metrics = None, **layers_kwargs):

    #necessary due to a bug
    #prev_data_format = tf.keras.backend.image_data_format()
    #tf.keras.backend.set_image_data_format('channels_last')
    
    layers = _RNNStack_layers(*layers_args, **layers_kwargs)
    model = tf.keras.models.Sequential(layers)

    #tf.keras.backend.set_image_data_format(prev_data_format)

    if optimizer is not None:
        optimizer = tf.keras.optimizers.get(optimizer)

    if loss is not None:
        loss = tf.keras.losses.get(loss)

    try:
        model.compile(optimizer = optimizer, loss = loss, metrics=metrics)
    except:
        pass

    return model

class RNN_TemporalUNet(tf.keras.models.Model):
    def __init__(self, input_units, grid_shape, out_channels = 1, rnn_config = None, temporalunet_config = None, nt = None):
        
        if rnn_config is None:
            rnn_config = {}
        rnn_final_units = int(np.prod(grid_shape))
        rnn_layers = _RNNStack_layers(input_units, rnn_final_units, nt = nt, **rnn_config)
        rnn_keras_tensors = convert_to_keras_tensors(rnn_layers)
        
        unet_input_shape = [nt,1,*grid_shape] if tf.keras.backend.image_data_format() == 'channels_first' else [nt,*grid_shape,1]
        unet_input_reshape_layer = tf.keras.layers.Reshape(unet_input_shape)(rnn_keras_tensors[-1])

        if temporalunet_config is None:
            temporalunet_config = {}
        unet_final_layer = TemporalUNet(input_layer = unet_input_reshape_layer, nt = None,  nx = None, ny = None, in_channels=None, out_channels=out_channels, padding='same', return_layer = True, **temporalunet_config)

        super().__init__(rnn_keras_tensors[0], unet_final_layer)

if __name__=='__main__':
    
    #tf.keras.backend.set_image_data_format('channels_first')


    nf = 16
    nx = 32
    ny = 128
    nt = 10
    nc = 2
    bsize = 5
    inp = tf.random.uniform((bsize,nt,nf))

    model = RNN_TemporalUNet(nf, [nx, ny], out_channels = nc, nt = nt)
