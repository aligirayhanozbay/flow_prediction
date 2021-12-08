import argparse
import json
import tensorflow as tf

tf.keras.backend.set_image_data_format('channels_first')

from ..models.SD_FNO import SD_FNO
from ..data.training.SpatiotemporalDataset import SpatiotemporalFRDataset
from ..losses import get as get_loss

parser = argparse.ArgumentParser()
parser.add_argument('experiment_config', type=str)
parser.add_argument('dataset_path', type=str)
parser.add_argument('checkpoint_path', type=str)
args = parser.parse_args()

config = json.load(open(args.experiment_config,'r'))

train_dataset, test_dataset, _ = SpatiotemporalFRDataset.from_hdf5(
    args.dataset_path,
    return_normalization_parameters=False,
    reshape_to_grid=True,
    retain_variable_dimension=True,
    **config['dataset'])
dummy_input, dummy_output = next(iter(train_dataset))
input_units = dummy_input.shape[-1]
n_timesteps = dummy_output.shape[1]
output_channels = dummy_output.shape[2]
grid_shape = [n_timesteps] + list(dummy_output.shape[3:])
del dummy_input, dummy_output

callbacks = [tf.keras.callbacks.ModelCheckpoint(args.checkpoint_path, save_best_only=True)]
if 'reduce_lr' in config['training']:
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**config['training']['reduce_lr']))
if 'early_stopping' in config['training']:
    callbacks.append(tf.keras.callbacks.EarlyStopping(**config['training']['early_stopping']))

#must use ReductionToOneDevice due to a tf bug
distribute_strategy =  tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.ReductionToOneDevice())
with distribute_strategy.scope():
    model = SD_FNO(
        input_units,
        grid_shape,
        out_channels=output_channels,
        time_and_space=True,
        **config['model']
    )
    model.summary()
    import pdb; pdb.set_trace()
    loss_fn = get_loss(config['training']['loss'])
    optim = tf.keras.optimizers.get(config['training']['optimizer'])
    metrics = config['training'].get('metrics', None)
    model.compile(loss=loss_fn, optimizer = optim, metrics = metrics)
    model.fit(train_dataset, epochs = config['training']['epochs'], validation_data = test_dataset, callbacks = callbacks)
    