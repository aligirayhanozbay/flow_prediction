import argparse
import json
import tensorflow as tf

from ..models.ShallowDecoder import original_shallow_decoder
from ..data.training.ShallowDecoderDataset import ShallowDecoderDataset

parser = argparse.ArgumentParser()
parser.add_argument('experiment_config', type=str)
parser.add_argument('dataset_path', type=str)
parser.add_argument('checkpoint_path', type=str)
args = parser.parse_args()

config = json.load(open(args.experiment_config,'r'))

train_dataset, test_dataset = ShallowDecoderDataset(args.dataset_path, **config['dataset'])
dummy_input, dummy_output = next(iter(train_dataset))
input_units = dummy_input.shape[-1]
output_units = dummy_output.shape[-1]
del dummy_input, dummy_output

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience = config['training']['reduce_lr_patience'], verbose=True),
    tf.keras.callbacks.ModelCheckpoint(args.checkpoint_path, save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience = config['training']['early_stopping_patience'], verbose=True)
]

distribute_strategy =  tf.distribute.MirroredStrategy()
with distribute_strategy.scope():
    model = original_shallow_decoder(input_units, output_units, learning_rate = config['training'].get('learning_rate', 1e-3), **config['model'])
    model.fit(train_dataset, epochs = config['training']['epochs'], validation_data = test_dataset, callbacks = callbacks)

