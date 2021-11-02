import argparse
import json
import tensorflow as tf
import h5py

tf.keras.backend.set_image_data_format('channels_first')

from .utils import get_normalization_var_order, get_mapping_params, get_grid_shape
from ..models.SD_UNet import SD_UNet_PI
from ..data.training.ShallowDecoderDataset import ShallowDecoderDataset
from ..losses.NS_annulus_residual_driver import NS_annulus_residual_driver

parser = argparse.ArgumentParser()
parser.add_argument('experiment_config', type=str)
parser.add_argument('dataset_path', type=str)
parser.add_argument('checkpoint_path', type=str)
args = parser.parse_args()

config = json.load(open(args.experiment_config,'r'))

h5file = h5py.File(args.dataset_path,'r')
mapping_params = get_mapping_params(h5file)
grid_shape = get_grid_shape(h5file)
PIloss = NS_annulus_residual_driver(mapping_params, reduce_mean = True, annulus_polar_coords = grid_shape, **config.get('loss', {}))
normalization_var_order = get_normalization_var_order(h5file)
output_var_order = config['dataset'].get('full_field_mask', normalization_var_order)

normalization = config['dataset'].get('normalization', None)
return_normalization_params = (normalization is not None)
full_field_mask = config['dataset'].pop('full_field_mask', normalization_var_order)
target_vars = list(set(['u','dudt','v','dvdt','p'] + full_field_mask))
train_dataset, test_dataset, case_id_map = ShallowDecoderDataset(args.dataset_path, retain_variable_dimension=True, reshape_to_grid=True, return_case_ids = True, return_normalization_parameters = return_normalization_params, return_case_names= False, full_field_mask = target_vars, **config['dataset'])

dummy_batch = next(iter(train_dataset))
dummy_input, dummy_output = dummy_batch[0], dummy_batch[1]
input_units = dummy_input.shape[-1]
output_channels = int((dummy_output.shape[1]/len(target_vars))*len(output_var_order))
del dummy_input, dummy_output, dummy_batch

callbacks = [tf.keras.callbacks.ModelCheckpoint(args.checkpoint_path, save_best_only=True)]
if 'reduce_lr' in config['training']:
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**config['training']['reduce_lr']))
if 'early_stopping' in config['training']:
    callbacks.append(tf.keras.callbacks.EarlyStopping(**config['training']['early_stopping']))

#distribute_strategy =  tf.distribute.MirroredStrategy()
#with distribute_strategy.scope():
optimizer = tf.keras.optimizers.get(config['training']['optimizer'])
model = SD_UNet_PI(
    input_units,
    grid_shape,
    PIloss,
    target_vars,
    output_var_order,
    out_channels = output_channels,
    data_normalization = config['dataset']['normalization'],
    normalization_var_order = normalization_var_order,
    optimizer = optimizer, metrics = config['training'].get('metrics', None),
    **config['model']
)
model.summary()
model.run_eagerly = True
model.fit(train_dataset, epochs = config['training']['epochs'], validation_data = test_dataset, callbacks = callbacks, validation_steps = config['training']['validation_steps'])
