import tensorflow as tf

from .UNet import slrelu

#from https://github.com/pomtojoer/flow_reconstruction

def dsc_ms(
        input_layer: tf.keras.layers.Layer = None,
        nx: int = None,
        ny: int = None,
        in_channels: int = 1,
        out_channels: int = None,
        data_format: str = None,
        activation = None,
        return_layer:bool=False):

    
    if data_format is not None:
        old_data_format = tf.keras.backend.image_data_format()
        tf.keras.backend.set_image_data_format(data_format)

    if activation is None:
        activation = 'relu'
    elif activation == 'leaky_relu':
        activation = tf.nn.leaky_relu
    elif activation == 'slrelu':
        activation = slrelu

    concat_axis = 1 if tf.keras.backend.image_data_format()=='channels_first' else -1
        
    #Model
    if input_layer is None:
        inpshape = (nx, ny, in_channels) if tf.keras.backend.image_data_format() == 'channels_last' else (in_channels,nx,ny)
        inputs = Input(shape=inpshape, name="inputs")
        input_img = inputs
    else:
        input_img = input_layer
        in_channels = int(input_layer.shape[concat_axis])

    if out_channels is None:
        out_channels = in_channels

    #Down sampled skip-connection model
    down_1 = tf.keras.layers.MaxPooling2D((8,8),padding='same')(input_img)
    x1 = tf.keras.layers.Conv2D(32, (3,3),activation=activation, padding='same')(down_1)
    x1 = tf.keras.layers.Conv2D(32, (3,3),activation=activation, padding='same')(x1)
    x1 = tf.keras.layers.UpSampling2D((2,2))(x1)

    down_2 = tf.keras.layers.MaxPooling2D((4,4),padding='same')(input_img)
    x2 = tf.keras.layers.Concatenate(axis=concat_axis)([x1,down_2])
    x2 = tf.keras.layers.Conv2D(32, (3,3),activation=activation, padding='same')(x2)
    x2 = tf.keras.layers.Conv2D(32, (3,3),activation=activation, padding='same')(x2)
    x2 = tf.keras.layers.UpSampling2D((2,2))(x2)

    down_3 = tf.keras.layers.MaxPooling2D((2,2),padding='same')(input_img)
    x3 = tf.keras.layers.Concatenate(axis=concat_axis)([x2,down_3])
    x3 = tf.keras.layers.Conv2D(32, (3,3),activation=activation, padding='same')(x3)
    x3 = tf.keras.layers.Conv2D(32, (3,3),activation=activation, padding='same')(x3)
    x3 = tf.keras.layers.UpSampling2D((2,2))(x3)

    x4 = tf.keras.layers.Concatenate(axis=concat_axis)([x3,input_img])
    x4 = tf.keras.layers.Conv2D(32, (3,3),activation=activation, padding='same')(x4)
    x4 = tf.keras.layers.Conv2D(32, (3,3),activation=activation, padding='same')(x4)

    #Multi-scale model (Du et al., 2018)
    layer_1 = tf.keras.layers.Conv2D(16, (5,5),activation=activation, padding='same')(input_img)
    x1m = tf.keras.layers.Conv2D(8, (5,5),activation=activation, padding='same')(layer_1)
    x1m = tf.keras.layers.Conv2D(8, (5,5),activation=activation, padding='same')(x1m)

    layer_2 = tf.keras.layers.Conv2D(16, (9,9),activation=activation, padding='same')(input_img)
    x2m = tf.keras.layers.Conv2D(8, (9,9),activation=activation, padding='same')(layer_2)
    x2m = tf.keras.layers.Conv2D(8, (9,9),activation=activation, padding='same')(x2m)

    layer_3 = tf.keras.layers.Conv2D(16, (13,13),activation=activation, padding='same')(input_img)
    x3m = tf.keras.layers.Conv2D(8, (13,13),activation=activation, padding='same')(layer_3)
    x3m = tf.keras.layers.Conv2D(8, (13,13),activation=activation, padding='same')(x3m)

    x_add = tf.keras.layers.Concatenate(axis=concat_axis)([x1m,x2m,x3m,input_img])
    x4m = tf.keras.layers.Conv2D(8, (7,7),activation=activation,padding='same')(x_add)
    x4m = tf.keras.layers.Conv2D(3, (5,5),activation=activation,padding='same')(x4m)

    x_final = tf.keras.layers.Concatenate(axis=concat_axis)([x4,x4m])
    x_final = tf.keras.layers.Conv2D(out_channels, (3,3),padding='same')(x_final)

    if data_format is not None:
        tf.keras.backend.set_image_data_format(old_data_format)

    if return_layer:
        return x_final
    else:
        model = tf.keras.models.Model(input_img, x_final)
        return model
