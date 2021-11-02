import numpy as np
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
import tensorflow as tf

from .ShallowDecoder import _get_original_shallow_decoder_layers
from .UNet import UNet
from .PILossModel import undo_mean_center

def convert_to_keras_tensors(layers_list):
    assert isinstance(layers_list[0], KerasTensor)
    keras_tensors = [layers_list[0]]
    for layer in layers_list[1:]:
        keras_tensors.append(layer(keras_tensors[-1]))
    return keras_tensors

class SD_UNet(tf.keras.models.Model):
    def __init__(self, input_units, grid_shape, out_channels = 1, sd_config = None, unet_config = None):
        
        if sd_config is None:
            sd_config = {}
        sd_final_units = int(np.prod(grid_shape))
        shallow_decoder_layers = _get_original_shallow_decoder_layers(input_units, sd_final_units, **sd_config)
        shallow_decoder_keras_tensors = convert_to_keras_tensors(shallow_decoder_layers)
        
        unet_input_shape = [1,*grid_shape] if tf.keras.backend.image_data_format() == 'channels_first' else [*grid_shape,1]
        unet_input_reshape_layer = tf.keras.layers.Reshape(unet_input_shape)(shallow_decoder_keras_tensors[-1])

        if unet_config is None:
            unet_config = {}
        unet_final_layer = UNet(input_layer = unet_input_reshape_layer, nx = None, ny = None, in_channels=None, out_channels=out_channels, padding='same', return_layer = True, **unet_config)

        super().__init__(shallow_decoder_keras_tensors[0], unet_final_layer)

class SD_UNet_PI(SD_UNet):
    _dummy_loss = tf.keras.losses.MeanSquaredError()
    _undo_normalization_functions = {
        'full_field_mean_center': undo_mean_center,
        'sensor_mean_center': undo_mean_center,
        None: None
    }
    _extract_piloss_transpose_indices = [1,0,2,3]
    _extract_other_loss_fwd_transpose_indices = [1,0,2,3]
    _extract_other_loss_backward_transpose_indices = [1,0,2,3]
    def __init__(self, input_units, grid_shape, *PILossargs, out_channels = 1, sd_config = None, unet_config = None, **PILosskwargs):
        super().__init__(input_units, grid_shape, out_channels = out_channels, sd_config = sd_config, unet_config = unet_config)

        #cannot do PI loss business with multiple inheritance due to TF bugs
        self._initialize_PI_loss(*PILossargs, **PILosskwargs)

    def _initialize_PI_loss(self, NS_annulus_residual_driver, target_var_order, output_var_order, other_loss = None, other_loss_weight = 1.0, PI_loss_weight = 1.0, optimizer = None, data_normalization = None, normalization_var_order = None, metrics = None):
        optimizer = optimizer if optimizer is not None else tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.compile(loss = self._dummy_loss, optimizer = optimizer, metrics = metrics)

        if isinstance(other_loss, str):
            other_loss = tf.keras.losses.get(other_loss)
        self.other_loss = other_loss
        self.other_loss_weight = other_loss_weight
        self.PI_loss_weight = PI_loss_weight

        self._target_var_indices = {v:k for k,v in enumerate(target_var_order)}
        self._n_target_vars = len(target_var_order)
        self._output_var_indices = {v:k for k,v in enumerate(output_var_order)}
        self._n_output_vars = len(output_var_order)

        self._undo_normalization = self._undo_normalization_functions[data_normalization]
        if data_normalization is not None:
            self._normalization_var_indices = {v:k for k,v in enumerate(normalization_var_order)}
        else:
            self._normalization_var_indices = None

        _target_indices_of_output_vars = []
        for var in self._output_var_indices:
            _target_indices_of_output_vars.append(self._target_var_indices[var])
        self._target_indices_of_output_vars = tf.constant(_target_indices_of_output_vars, dtype=tf.int32)
        
        self.PILoss = NS_annulus_residual_driver
        self.annulus_grid_shape = self.PILoss.residual_losses[0].annulus_polar_coords.shape[:2]

        PILoss_target_extraction_indices = []
        PILoss_output_extraction_indices = []
        PILoss_target_scatter_indices = []
        PILoss_output_scatter_indices = []
        for var_idx,var in enumerate(['u','dudt','v','dvdt','p']):
            if var in self._output_var_indices:
                PILoss_output_extraction_indices.append(self._output_var_indices[var])
                PILoss_output_scatter_indices.append(var_idx)
            else:
                PILoss_target_extraction_indices.append(self._target_var_indices[var])
                PILoss_target_scatter_indices.append(var_idx)
        self._PILoss_target_extraction_indices = tf.constant(PILoss_target_extraction_indices, tf.int32)
        self._PILoss_output_extraction_indices = tf.constant(PILoss_output_extraction_indices, tf.int32)
        self._PILoss_stacking_reorder_indices = tf.constant(PILoss_output_scatter_indices + PILoss_target_scatter_indices, tf.int32)

    def reshape_for_other_loss(self, x, *args, **kwargs):
        return x

    def reshape_for_PILoss(self, x, *args, **kwargs):
        return x

    def extract_PILoss_inputs(self, pred, targets, normalization_params = None):
        pred = self.reshape_for_PILoss(pred, for_target = False)
        targets = self.reshape_for_PILoss(targets, for_target = True)

        if normalization_params is not None:
            targets = self._undo_normalization(targets, normalization_params)
            pred = self._undo_normalization(pred, normalization_params)
        
        targets_t = tf.transpose(targets, self._extract_piloss_transpose_indices)
        pred_t = tf.transpose(pred, self._extract_piloss_transpose_indices)
        vars_from_target = tf.gather(targets_t, self._PILoss_target_extraction_indices)
        vars_from_pred = tf.gather(pred_t, self._PILoss_output_extraction_indices)
        PILoss_inputs = tf.concat([vars_from_pred, vars_from_target],0)
        PILoss_inputs = tf.gather(PILoss_inputs, self._PILoss_stacking_reorder_indices)
        PILoss_inputs = tf.unstack(tf.expand_dims(PILoss_inputs, 2), num=5, axis=0)
        
        return PILoss_inputs

    def extract_other_loss_inputs(self, targets, pred):
        pred = self.reshape_for_other_loss(pred, for_target = False)
        targets = self.reshape_for_other_loss(targets, for_target = True)
        #import pdb; pdb.set_trace()
        targets = tf.gather(tf.transpose(targets,self._extract_other_loss_fwd_transpose_indices),self._target_indices_of_output_vars)
        targets = tf.transpose(targets, self._extract_other_loss_backward_transpose_indices)

        return targets, pred

    def train_step(self, batch):
        if self._undo_normalization is None:
            inputs, targets, case_ids = batch
        else:
            inputs, targets, normalization_params, case_ids = batch
        #import pdb; pdb.set_trace()
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)

            pred = self(inputs)

            other_loss_inputs = self.extract_other_loss_inputs(targets, pred)
            other_loss_val = tf.reduce_mean(self.other_loss(*other_loss_inputs))

            #reshape the prediction to NCHW format for the physics informed loss
            PILoss_inputs = self.extract_PILoss_inputs(pred, targets)
            PILoss_val = self.PILoss(*PILoss_inputs, case_ids)
            #import pdb; pdb.set_trace()
            loss_val = self.other_loss_weight * other_loss_val + self.PI_loss_weight * PILoss_val

        grads = tape.gradient(loss_val, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {'loss': loss_val, 'other_loss': other_loss_val, 'pi_loss': PILoss_val}

    def test_step(self, batch):
        if self._undo_normalization is None:
            inputs, targets, case_ids = batch
        else:
            inputs, targets, normalization_params, case_ids = batch

        pred = self(inputs)

        other_loss_inputs = self.extract_other_loss_inputs(targets, pred)
        other_loss_val = tf.reduce_mean(self.other_loss(*other_loss_inputs))

        #reshape the prediction to NCHW format for the physics informed loss
        PILoss_inputs = self.extract_PILoss_inputs(pred, targets)
        PILoss_val = self.PILoss(*PILoss_inputs, case_ids)

        
        loss_val = self.other_loss_weight * other_loss_val + self.PI_loss_weight * PILoss_val

        return {'loss': loss_val, 'other_loss': other_loss_val, 'pi_loss': PILoss_val}

if __name__ == '__main__':

    mod = SD_UNet(14,(64,256),out_channels = 3)
    mod.summary()
