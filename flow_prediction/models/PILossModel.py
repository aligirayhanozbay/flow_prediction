import tensorflow as tf
import warnings

from ..losses.NS_annulus_residual_driver import NS_annulus_residual_driver as NS_annulus_residual_driver_cls

def undo_mean_center(x, means):
    return x + means

class PILossModel:
    _dummy_loss = tf.keras.losses.MeanSquaredError()
    _undo_normalization_functions = {
        'full_field_mean_center': undo_mean_center,
        'sensor_mean_center': undo_mean_center,
        None: None
    }
    _extract_piloss_transpose_indices = [1,0,2,3]
    _extract_other_loss_fwd_transpose_indices = [2,0,1]
    _extract_other_loss_backward_transpose_indices = [1,2,0]
    def __init__(self, NS_annulus_residual_driver, target_var_order, output_var_order, other_loss = None, other_loss_weight = 1.0, PI_loss_weight = 1.0, optimizer = None, data_normalization = None, normalization_var_order = None, metrics = None):
        if not isinstance(NS_annulus_residual_driver, NS_annulus_residual_driver_cls):
            warnings.warn('Wrong class instance provided for NS_annulus_residual_driver. Currently, due to a strange interaction with TF, __init__ returns immediately when this happens.')
            return
        
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


    def reshape_for_other_loss(self, x, for_target = False):
        nvars = self._n_target_vars if for_target else self._n_output_vars
        bsize = tf.shape(x)[0]
        x = tf.reshape(x, [bsize, -1, nvars])
        return x
        
    def reshape_for_PILoss(self, x, for_target = False):
        bsize = tf.shape(x)[0]
        x = self.reshape_for_other_loss(x, for_target = for_target)
        nvars = tf.shape(x)[-1]
        x = tf.transpose(x, [0,2,1])
        x = tf.reshape(x, [bsize, nvars, *self.annulus_grid_shape])
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
        
        # PIloss_inputs = []
        # for var in ['u','dudt','v','dvdt','p']:
        #     if var in self._output_var_indices:
        #         var_value = pred[:,self._output_var_indices[var],...]
        #     else:
        #         var_value = targets[:,self._target_var_indices[var],...]
        #     PIloss_inputs.append(tf.expand_dims(var_value,1))
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
