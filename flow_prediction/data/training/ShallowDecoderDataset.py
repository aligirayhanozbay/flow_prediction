import h5py
import numpy as np
import itertools
import warnings
import copy
import tensorflow as tf

def _extract_variable_names(h5file,nvars=None):
    variable_ids = list(filter(lambda x: 'variable' in x, sorted(h5file.attrs.keys())))
    if nvars is None:
        nvars = h5file[list(h5file.keys())[0]]['sensor_values'].shape[-1]
        
    if nvars == len(variable_ids):
        return [h5file.attrs[varid].decode('utf-8') for varid in variable_ids]
    else:
        warnings.warn('Dataset file has fewer tags for variable names than there are field variables.')
        return None

def _get_unique_variable_mask_ids(sensor_mask, variable_name_idx_map=None):
    result = []
    for s in sensor_mask:
        if isinstance(s,str):
            result.append(variable_name_idx_map[s])
        elif isinstance(s,int):
            result.append(s)
        else:
            raise(ValueError('values in sensor_mask must be strings present in variable_names or ints'))
    return sorted(list(set(result)))
        

def _sensor_mean_center(sensor_values, full_field_values):
    means = np.mean(sensor_values, axis=1, keepdims=True)
    return sensor_values-means, full_field_values-means, np.squeeze(means)

def _fullfield_mean_center(sensor_values, full_field_values):
    means = np.mean(full_field_values, axis=1, keepdims=True)
    return sensor_values-means, full_field_values-means, np.squeeze(means)

def _extract_values(case, sensor_slice=None, full_field_slice=None, normalization=None):
    normalization_methods = {'sensor_mean_center': _sensor_mean_center, 'full_field_mean_center': _fullfield_mean_center, None: lambda x,y: (x,y,np.zeros(x.shape[:1],dtype=x.dtype))}
    if sensor_slice is None:
        sensor_slice = slice(0,None,None)
    if full_field_slice is None:
        full_field_slice = slice(0,None,None)
    ntimesteps = case['sensor_values'].shape[0]
    sensor_values, full_field_values, normalization_params = normalization_methods[normalization](np.array(case['sensor_values']), np.array(case['full_field_values']))
    sensor_values = sensor_values.reshape(ntimesteps,-1)[:,sensor_slice]
    full_field_values = full_field_values[...,full_field_slice].reshape(ntimesteps,-1)
    return sensor_values, full_field_values, normalization_params

def ShallowDecoderDataset(h5path, batch_size=None, shuffle=False, retain_time_dimension=False, sensor_masks = None, full_field_mask = None, h5py_driver = None, h5py_driver_kwargs = None, normalization = None, dtypes = None, train_test_split = None, rand_seed = None, return_case_names = False, return_normalization_parameters = False):
    if train_test_split is not None:
        assert ((1.0-train_test_split) > 0.0)
        
    h5file = h5py.File(h5path, 'r', driver = h5py_driver, **({} if h5py_driver_kwargs is None else h5py_driver_kwargs))

    if dtypes is None:
        dtypes = ['float32','float32','float32']
    elif (isinstance(dtypes,list) or isinstance(dtypes,tuple)) and len(dtypes) == 2:
        dtypes = [dtypes[0],dtypes[1],dtypes[0]]
    elif (not isinstance(dtypes,list)) or (not isinstance(dtypes,tuple)):
        dtypes = [dtypes,dtypes,dtypes]

    example_case = h5file[list(h5file.keys())[0]]
    nvars = example_case['sensor_values'].shape[-1]

    variable_names = _extract_variable_names(h5file, nvars=nvars)

    if variable_names is not None:
        variable_name_idx_map = {varname:k for k, varname in enumerate(variable_names)}
    else:
        variable_name_idx_map = None

    if sensor_masks is None:
        sensor_slice = sensor_masks
    else:
        nsensors = example_case['sensor_values'].shape[-2]
        assert nsensors == len(sensor_masks)
        #get indices of variables that will be used for each sensor. e.g. if variable_names are ['p','u','v'] which are stored across indices 0,1,2 of last dim of case['sensor_values'], and sensor_masks is [['u','v'],['p'],['p','u','v']], variable_indices will be [[1,2],[0],[0,1,2]]
        variable_indices = [_get_unique_variable_mask_ids(sensor_mask, variable_name_idx_map) for sensor_mask in sensor_masks]
        n_variables_per_sensor = np.array([len(indices) for indices in variable_indices])
        #build an array which represents which row each element in the flattened version of variable_indices used to belong to
        sensor_indices = np.repeat(np.arange(nsensors), n_variables_per_sensor)
        sensor_local_variable_indices = np.array(list(itertools.chain.from_iterable(variable_indices)))
        #for the variable_indices example above, case['sensor_values'] would have an array of shape (3,3) for each time slice. global_variable_indices represents the index of each item in variable_indices in the flattened version of case['sensor_values'][t]
        global_variable_indices = sensor_indices*nvars + sensor_local_variable_indices
        sensor_slice = global_variable_indices

    if full_field_mask is None:
        full_field_slice = full_field_slice
    else:
        ff_variable_indices = _get_unique_variable_mask_ids(full_field_mask, variable_name_idx_map)
        full_field_slice = np.array(ff_variable_indices)
        
    sensor_values, full_field_values, normalization_params = zip(*[_extract_values(h5file[case], sensor_slice = sensor_slice, full_field_slice=full_field_slice, normalization=normalization) for case in h5file.keys()])
    case_names = np.concatenate([[case_name for _ in range(h5file[case_name]['sensor_values'].shape[0])] for case_name in h5file.keys()],0)

    if retain_time_dimension:
        sensor_values = np.stack(sensor_values,0)
        full_field_values = np.stack(full_field_values,0)
        normalization_params = np.stack(normalization_params,0)
    else:
        sensor_values = np.concatenate(sensor_values,0)
        full_field_values = np.concatenate(full_field_values,0)
        normalization_params = np.concatenate(normalization_params,0)
    
    sensor_values = sensor_values.astype(dtypes[0])
    full_field_values = full_field_values.astype(dtypes[1])
    normalization_params = normalization_params.astype(dtypes[2])
    
    if train_test_split is None:
        args_list = [sensor_values, full_field_values]
        if return_normalization_parameters:
            args_list.append(normalization_params)
        if return_case_names:
            args_list.append(case_names)
        dset = tf.data.Dataset.from_tensor_slices(tuple(args_list))
        if shuffle:
            dset = dset.shuffle()
        if batch_size is not None:
            dset = dset.batch(batch_size)
        return dset
    else:
        if rand_seed is not None:
            np.random.seed(rand_seed)
        global_indices = np.arange(sensor_values.shape[0])
        shuffled_indices = copy.deepcopy(global_indices)
        np.random.shuffle(shuffled_indices)
        max_test_index = round(sensor_values.shape[0] * (1.0-train_test_split))
        test_indices = shuffled_indices[:max_test_index]
        train_indices = shuffled_indices[max_test_index:]

        if not shuffle:
            train_indices = np.sort(train_indices)
            test_indices = np.sort(test_indices)

        train_sensor = sensor_values[train_indices]
        train_fullfield = full_field_values[train_indices]
        train_normalization_params = normalization_params[train_indices]
        train_case_names = case_names[train_indices]
        
        test_sensor = sensor_values[test_indices]
        test_fullfield = full_field_values[test_indices]
        test_normalization_params = normalization_params[test_indices]
        test_case_names = case_names[test_indices]

        train_args_list = [train_sensor, train_fullfield]
        test_args_list = [test_sensor, test_fullfield]
        if return_normalization_parameters:
            train_args_list.append(train_normalization_params)
            test_args_list.append(test_normalization_params)
        if return_case_names:
            train_args_list.append(train_case_names)
            test_args_list.append(test_case_names)

        dset_train = tf.data.Dataset.from_tensor_slices(tuple(train_args_list))
        dset_test = tf.data.Dataset.from_tensor_slices(tuple(test_args_list))

        if batch_size is not None:
            dset_train = dset_train.batch(batch_size)
            dset_test = dset_test.batch(batch_size)

        return dset_train, dset_test
        
        
    

if __name__ == '__main__':
    fname = '/storage/pyfr/meshes_maprun2/postprocessed.h5'
    sensor_masks = list(itertools.chain.from_iterable([[['p'] for _ in range(10)],[['u','v'] for _ in range(4)]]))
    full_field_mask = ['u','v']
    retain_time_dimension = False
    normalization = 'sensor_mean_center'
    ShallowDecoderDataset(fname, sensor_masks = sensor_masks, full_field_mask = full_field_mask, retain_time_dimension = retain_time_dimension, normalization = normalization)
    
    
    
    
    
