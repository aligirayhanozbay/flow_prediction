import argparse
import json
import tensorflow as tf
import h5py
import itertools
import re

tf.keras.backend.set_image_data_format('channels_first')

from ..models.FNO import FNO
from ..models.SD_UNet import SD_UNet
from ..models.SD_FNO import SD_FNO
from ..models.ShallowDecoder import original_shallow_decoder
from ..data.training.TimeMarchingDataset import TimeMarchingDataset
from ..losses import get as get_loss

parser = argparse.ArgumentParser()
parser.add_argument('experiment_config', type=str)
parser.add_argument('dataset_path', type=str)
parser.add_argument('checkpoint_path', type=str)
parser.add_argument('--reconstruction_model_weights', type=str, default=None)
args = parser.parse_args()

config = json.load(open(args.experiment_config,'r'))

with h5py.File(args.dataset_path, 'r') as df:
    grid_shape = list(df.attrs['grid_shape'])
    try:
        output_channels = len(config['dataset']['full_field_mask'])
    except:
        output_channels = df[list(df.keys())[0]]['full_field_values'].shape[-1]
    try:
        input_units = len(list(itertools.chain.from_iterable(config['dataset']['sensor_masks'])))
    except:
        input_units = int(tf.reduce_prod(df[list(df.keys())[0]]['sensor_values'].shape[1:]))

if (not ('dataset_kwargs' in config['dataset'])):
    config['dataset']['dataset_kwargs'] = {}
config['dataset']['dataset_kwargs']['return_sensor_values']=False
    
if args.reconstruction_model_weights is not None and ('reconstruction_model' in config):
    model_type = config.get('reconstruction_model_type', 'SD-UNet')
    if model_type == 'SD-UNet':
        rcm = SD_UNet(input_units, grid_shape, out_channels = output_channels, **config['reconstruction_model'])
        rcm.load_weights(args.reconstruction_model_weights)
        print('Successfully loaded ' + args.reconstruction_model_weights)
    elif model_type == 'SD-FNO':
        rcm = SD_FNO(input_units, grid_shape, out_channels = output_channels, **config['reconstruction_model'])
        rcm.load_weights(args.reconstruction_model_weights)
        print('Successfully loaded ' + args.reconstruction_model_weights)
    elif model_type == 'SD':
        output_units = int(tf.reduce_prod(tf.concat([[output_channels],grid_shape],0)))
        rcm = original_shallow_decoder(input_units, output_units, **config['reconstruction_model'])
        rcm.load_weights(args.reconstruction_model_weights)
        print('Successfully loaded ' + args.reconstruction_model_weights)
        rcm = tf.keras.Sequential([tf.keras.layers.Input(input_units), rcm, tf.keras.layers.Reshape((output_channels, *grid_shape))])
    config['dataset']['dataset_kwargs']['reconstruction_model'] = rcm

train_dataset, test_dataset, _ = TimeMarchingDataset.from_hdf5(
    args.dataset_path,
    return_normalization_parameters=False,
    return_sensor_values=False,
    return_case_ids=False,
    return_case_names=False,
    reshape_to_grid=True,
    retain_variable_dimension=True,
    **config['dataset'])
del rcm

callbacks = [tf.keras.callbacks.ModelCheckpoint(args.checkpoint_path, save_best_only=True, save_weights_only=True)]
if 'reduce_lr' in config['training']:
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**config['training']['reduce_lr']))
if 'early_stopping' in config['training']:
    callbacks.append(tf.keras.callbacks.EarlyStopping(**config['training']['early_stopping']))

distribute_strategy =  tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
with distribute_strategy.scope():
    model = FNO([output_channels, *grid_shape], output_channels, return_layer=False, **config['model'])
    model.summary()
    loss_fn = get_loss(config['training']['loss'])
    model.compile(loss=loss_fn,
        optimizer = tf.keras.optimizers.get(config['training']['optimizer']),
        metrics = config['training'].get('metrics', None))
    model.fit(train_dataset, epochs = config['training']['epochs'], validation_data = test_dataset, callbacks = callbacks)
