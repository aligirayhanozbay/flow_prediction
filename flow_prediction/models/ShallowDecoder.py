import tensorflow as tf

def original_shallow_decoder(input_layer_shape, output_layer_size, learning_rate, hidden_layer_units = None, hidden_layer_activations = None, normalization = None, l2_regularization = 0.0, loss_function = None, metrics = None):

    if hidden_layer_units is None:
        hidden_layer_units = [40,40]
    if hidden_layer_activations is None:
        hidden_layer_activations = ['relu' for _ in hidden_layer_units]
    assert len(hidden_layer_activations) == len(hidden_layer_units)

    normalization_map = {'batchnorm': tf.keras.layers.BatchNormalization, 'layernorm': tf.keras.layers.LayerNormalization}

    layers = [tf.keras.layers.Input(input_layer_shape)]
    for units, activation in zip(hidden_layer_units, hidden_layer_activations):
        regularizer = tf.keras.regularizers.L2(l2_regularization) if abs(l2_regularization) > 0.0 else None
        layers.append(tf.keras.layers.Dense(units, activation = activation, kernel_initializer='glorot_normal', kernel_regularizer = regularizer, bias_regularizer = regularizer))
        if normalization is not None:
            layers.append(normalization_map[normalization]())
    layers.append(tf.keras.layers.Dense(output_layer_size))

    model = tf.keras.models.Sequential(layers)
    
    # Model
    # model = tf.keras.models.Sequential([
    #     # Layer 1
    #     tf.keras.layers.Input(input_layer_shape),
    #     tf.keras.layers.Dense(40, activation='relu', kernel_initializer='glorot_normal'),
    #     tf.keras.layers.BatchNormalization(),
        
    #     # Layer 2
    #     tf.keras.layers.Dense(45, activation='relu', kernel_initializer='glorot_normal'),
    #     tf.keras.layers.BatchNormalization(),
        
    #     # Output layer
    #     tf.keras.layers.Dense(output_layer_size),
    # ])

    # Optimiser
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Defining losses
    if loss_function is None:
        loss_function = tf.keras.losses.MeanSquaredError()

    # Defining metrics
    #metrics = tf.keras.metrics.MeanAbsoluteError()

    model.compile(optimizer=optimiser, loss=loss_function, metrics=metrics)

    return model

if __name__ == '__main__':
    m = original_shallow_decoder(18, 20000, 1e-4, normalization='layernorm')
    m.summary()
    