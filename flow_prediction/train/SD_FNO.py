import argparse
import json
import tensorflow as tf

tf.keras.backend.set_image_data_format('channels_first')

from ..models.SD_FNO import SD_FNO
from ..data.training.ShallowDecoderDataset import ShallowDecoderDataset
from ..losses import get as get_loss

parser = argparse.ArgumentParser()
parser.add_argument('experiment_config', type=str)
parser.add_argument('dataset_path', type=str)
parser.add_argument('checkpoint_path', type=str)
args = parser.parse_args()

config = json.load(open(args.experiment_config,'r'))

train_dataset, test_dataset = ShallowDecoderDataset(args.dataset_path, retain_variable_dimension=True, reshape_to_grid=True, **config['dataset'])
dummy_input, dummy_output = next(iter(train_dataset))
input_units = dummy_input.shape[-1]
output_channels = dummy_output.shape[1]
grid_shape = dummy_output.shape[2:]
del dummy_input, dummy_output

callbacks = [tf.keras.callbacks.ModelCheckpoint(args.checkpoint_path, save_best_only=True, save_weights_only=True)]
if 'reduce_lr' in config['training']:
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**config['training']['reduce_lr']))
if 'early_stopping' in config['training']:
    callbacks.append(tf.keras.callbacks.EarlyStopping(**config['training']['early_stopping']))

#cannot use complex weight directly due to tf bug - cant use both complex and float weights
#in a model when using tf.distribute.MirroredStrategy...
# distribute_strategy =  tf.distribute.MirroredStrategy()
# with distribute_strategy.scope():
model = SD_FNO(input_units, grid_shape, out_channels = output_channels, **config['model'])
model.summary()
loss_fn = get_loss(config['training']['loss'])
model.compile(loss=loss_fn, optimizer = tf.keras.optimizers.get(config['training']['optimizer']), metrics = config['training'].get('metrics', None))
model.fit(train_dataset, epochs = config['training']['epochs'], validation_data = test_dataset, callbacks = callbacks)
