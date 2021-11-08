import tensorflow as tf
import copy

from .AttentionMeanLpLoss import AttentionMeanLpLoss

def get(spec):
    if isinstance(spec, dict) and ('class_name' in spec) and (spec['class_name'] == 'AttentionMeanLpLoss'):
        nspec = copy.deepcopy(spec)
        _ = nspec.pop('class_name')
        return AttentionMeanLpLoss(**nspec)
    else:
        return tf.keras.losses.get(spec)
    
