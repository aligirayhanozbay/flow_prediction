import numpy as np

def get_normalization_var_order(f):
    attr_keys = list(f.attrs.keys())
    varnames = [f.attrs[k].decode('utf-8') for k in filter(lambda key: 'variable' in key, attr_keys)]
    return varnames

def get_grid_shape(f):
    return list(f.attrs['grid_shape'])

def get_mapping_params(f):
    case_indices = {case:k for k,case in enumerate(f.keys())}
    return [{'annulusmap':{param:np.array(f[case]['amap'][param]) for param in f[case]['amap']}} for case in case_indices]
