import argparse
import json
import tensorflow as tf
import h5py
import numpy as np

from ..models.ShallowDecoderPI import ShallowDecoderPI
from ..data.training.ShallowDecoderDataset import ShallowDecoderDataset
from ..losses.NS_annulus_residual_driver import NS_annulus_residual_driver

def get_normalization_var_order(f):
    attr_keys = list(f.attrs.keys())
    varnames = [f.attrs[k].decode('utf-8') for k in attr_keys]
    return varnames

def get_mapping_params(f):
    case_indices = {case:k for k,case in enumerate(f.keys())}
    return [{'annulusmap':{param:np.array(f[case]['amap'][param]) for param in f[case]['amap']}} for case in case_indices]
    

parser = argparse.ArgumentParser()
parser.add_argument('experiment_config', type=str)
parser.add_argument('dataset_path', type=str)
parser.add_argument('checkpoint_path', type=str)
parser.add_argument('--load_checkpoint', type=str, default=None)
args = parser.parse_args()

config = json.load(open(args.experiment_config,'r'))

h5file = h5py.File(args.dataset_path,'r')
mapping_params = get_mapping_params(h5file)
PIloss = NS_annulus_residual_driver(mapping_params, reduce_mean = True, **config.get('loss', {}))
normalization_var_order = get_normalization_var_order(h5file)
output_var_order = config['dataset'].get('full_field_mask', normalization_var_order)

normalization = config['dataset'].get('normalization', None)
return_normalization_params = (normalization is not None)
full_field_mask = config['dataset'].pop('full_field_mask', normalization_var_order)
target_vars = list(set(['u','dudt','v','dvdt','p'] + full_field_mask))
train_dataset, test_dataset, case_id_map = ShallowDecoderDataset(args.dataset_path, return_case_ids = True, return_normalization_parameters = return_normalization_params, return_case_names = False, full_field_mask = target_vars, **config['dataset'])
dummy_batch = next(iter(train_dataset))
input_units = dummy_batch[0].shape[-1]
output_units = int((dummy_batch[1].shape[-1]/len(target_vars))*len(output_var_order))
del dummy_batch

callbacks = [tf.keras.callbacks.ModelCheckpoint(args.checkpoint_path, save_best_only=True)]
if 'reduce_lr' in config['training']:
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**config['training']['reduce_lr']))
if 'early_stopping' in config['training']:
    callbacks.append(tf.keras.callbacks.EarlyStopping(**config['training']['early_stopping']))

#distribute_strategy =  tf.distribute.MirroredStrategy()
#with distribute_strategy.scope():
optimizer = tf.keras.optimizers.Adam(learning_rate = config['training'].get('learning_rate', 1e-3))
model = ShallowDecoderPI(PIloss, target_vars, output_var_order, input_units, output_units, optimizer=optimizer, data_normalization=config['dataset']['normalization'], normalization_var_order = normalization_var_order, **config['model'])
if args.load_checkpoint is not None:
    model.load_weights(args.load_checkpoint)
model.run_eagerly = True
model.fit(train_dataset, epochs = config['training']['epochs'], validation_data = test_dataset, callbacks = callbacks, validation_steps = config['training'].get('validation_steps', None))
