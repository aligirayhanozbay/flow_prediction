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
        return [h5file.attrs[varid] for varid in variable_ids]
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

def _extract_case_values(case, sensor_slice=None, full_field_slice=None, normalization=None, retain_variable_dimension=False, var_shape = None):
    normalization_methods = {'sensor_mean_center': _sensor_mean_center, 'full_field_mean_center': _fullfield_mean_center, None: lambda x,y: (x,y,np.zeros(x.shape[:1],dtype=x.dtype))}
    if sensor_slice is None:
        sensor_slice = slice(0,None,None)
    if full_field_slice is None:
        full_field_slice = slice(0,None,None)
    ntimesteps = case['sensor_values'].shape[0]
    sensor_values, full_field_values, normalization_params = normalization_methods[normalization](np.array(case['sensor_values']), np.array(case['full_field_values']))
    sensor_values = sensor_values.reshape(ntimesteps,-1)[:,sensor_slice]
    full_field_values = full_field_values[...,full_field_slice]

    if not retain_variable_dimension:
        full_field_values = full_field_values.reshape(ntimesteps,-1)
    elif var_shape is not None:
        newshape = [ntimesteps,*var_shape,full_field_values.shape[-1]]
        full_field_values = full_field_values.reshape(newshape)
        if tf.keras.backend.image_data_format() == 'channels_first':
            transpose_ord = [0,len(newshape)-1] + list(range(1,len(newshape)-1))
            full_field_values = full_field_values.transpose(transpose_ord)
            
    return sensor_values, full_field_values, normalization_params

def _extract_values(h5path, retain_time_dimension=False, retain_variable_dimension=False, reshape_to_grid=False, sensor_masks = None, full_field_mask = None, h5py_driver = None, h5py_driver_kwargs = None, normalization = None, dtypes = None):
        
    h5file = h5py.File(h5path, 'r', driver = h5py_driver, **({} if h5py_driver_kwargs is None else h5py_driver_kwargs))
    case_id_map = {case:k for k,case in enumerate(h5file.keys())}

    if reshape_to_grid:
        grid_shape = h5file.attrs['grid_shape']
    else:
        grid_shape = None

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
        full_field_slice = None
    else:
        ff_variable_indices = _get_unique_variable_mask_ids(full_field_mask, variable_name_idx_map)
        full_field_slice = np.array(ff_variable_indices)
        
    sensor_values, full_field_values, normalization_params = zip(*[_extract_case_values(h5file[case], sensor_slice = sensor_slice, full_field_slice=full_field_slice, normalization=normalization, retain_variable_dimension=retain_variable_dimension, var_shape = grid_shape) for case in h5file.keys()])
    case_names = [[case_name for _ in range(h5file[case_name]['sensor_values'].shape[0])] for case_name in h5file.keys()]

    if retain_time_dimension:
        sensor_values = np.stack(sensor_values,0)
        full_field_values = np.stack(full_field_values,0)
        normalization_params = np.stack(normalization_params,0)
        case_names = np.stack(case_names,0)
        case_ids = np.vectorize(lambda x: np.vectorize(lambda y: case_id_map[y])(x))(case_names)
    else:
        sensor_values = np.concatenate(sensor_values,0)
        full_field_values = np.concatenate(full_field_values,0)
        normalization_params = np.concatenate(normalization_params,0)
        case_names = np.concatenate(case_names,0)
        case_ids = np.vectorize(lambda x: case_id_map[x])(case_names)
    
    sensor_values = sensor_values.astype(dtypes[0])
    full_field_values = full_field_values.astype(dtypes[1])
    normalization_params = normalization_params.astype(dtypes[2])

    h5file.close()
    
    return sensor_values, full_field_values, normalization_params, case_names, case_ids, case_id_map

def ShallowDecoderDataset(h5path, batch_size=None, shuffle=False, train_test_split = None, train_test_split_by_case = False, rand_seed = None, return_case_names = False, return_case_ids = False, return_normalization_parameters = False, **extraction_kwargs):
    '''
    Generates tf.data.Dataset objects for training Shallow Decoder models.

    Inputs:
    -h5path: str. Path to the hdf5 file containing the data, generated via flow_prediction.data.utils.postprocess.
    -batch_size: int.
    -shuffle: bool. Enable/disable shuffling.
    -retain_time_dimension: bool. If True, inputs/outputs will have a dummy (size 1) time dimension.
    -retain_variable_dimension: bool. If True, inputs/outputs will have an extra dim. for every variable such as u or v. If False, all variables will be concatenated to a single dim. Useful for having a separate channel for conv model inputs. Use alongside reshape_to_grid.
    -reshape_to_grid: bool. If True, all outputs will have the grid shape contained in the hdf5 file's grid shape attribute.
    -sensor_masks: List[List[str]]. List of variable names to keep for the inputs per sensor - e.g. [['u','p'],['v']] means the inputs corresponding to the 1st sensor will only contain u vel. and pressure, and the 2nd will only have v vel.
    -full_field_mask: List[str]. Similar to sensor_masks, but for the whole high-fidelity field. 
    -h5py_driver: str. See h5py documentation.
    -h5py_driver_kwargs: Dict. See h5py documentation.
    -normalization: str. 'sensor_mean_center', 'full_field_mean_center' or None.
    -dtypes: List[str]. Numpy data types for the inputs, outputs and normalization parameters.
    -train_test_split: float. Proportion of data to allocate to the training set.
    -train_test_split_by_case: bool. If True, train/test split is done on a case basis so the training set and test set have snapshots from different cases.
    -rand_seed: int. See numpy documentation.
    -return_case_names: bool. If True, each tf.data.Dataset object will also return info about the name of the case each sample is from.
    -return_case_ids: bool. Similar to return_case_names, but uses autogenerated integer ids instead.
    -return_normalization_parameters: bool. If True, returns the normalization parameters associated with each sample.
    '''
    if train_test_split is not None:
        assert ((1.0-train_test_split) > 0.0)
        
    sensor_values, full_field_values, normalization_params, case_names, case_ids, case_id_map = _extract_values(h5path, **extraction_kwargs)

    return_list = []
    if train_test_split is None:
        args_list = [sensor_values, full_field_values]
        if return_normalization_parameters:
            args_list.append(normalization_params)
        if return_case_names:
            args_list.append(case_names)
        if return_case_ids:
            args_list.append(case_ids)
        dset = tf.data.Dataset.from_tensor_slices(tuple(args_list))
        if shuffle:
            dset = dset.shuffle()
        if batch_size is not None:
            dset = dset.batch(batch_size)
        return_list.append(dset)
    else:
        if rand_seed is not None:
            np.random.seed(rand_seed)
        global_indices = np.arange(sensor_values.shape[0])
        
        if train_test_split_by_case:
            max_test_case_id = round(len(case_id_map.keys()) * (1.0-train_test_split))
            test_case_mask = case_ids < max_test_case_id
            test_indices = np.where(test_case_mask)[0]
            train_indices = np.where(np.logical_not(test_case_mask))[0]
            if shuffle:
                np.random.shuffle(train_indices)
                np.random.shuffle(test_indices)
        else:
            max_test_index = round(sensor_values.shape[0] * (1.0-train_test_split))
            shuffled_indices = copy.deepcopy(global_indices)
            np.random.shuffle(shuffled_indices)
            test_indices = shuffled_indices[:max_test_index]
            train_indices = shuffled_indices[max_test_index:]
            if not shuffle:
                #necessary to shuffle and then sort if not splitting by case to ensure i.i.d. split
                test_indices = global_indices[:max_test_index]
                train_indices = global_indices[max_test_index:]

        #import pdb; pdb.set_trace()

        train_sensor = sensor_values[train_indices]
        train_fullfield = full_field_values[train_indices]
        
        test_sensor = sensor_values[test_indices]
        test_fullfield = full_field_values[test_indices]

        train_args_list = [train_sensor, train_fullfield]
        test_args_list = [test_sensor, test_fullfield]
        if return_normalization_parameters:
            train_normalization_params = normalization_params[train_indices]
            test_normalization_params = normalization_params[test_indices]
            train_args_list.append(train_normalization_params)
            test_args_list.append(test_normalization_params)
        if return_case_names:
            train_case_names = case_names[train_indices]
            test_case_names = case_names[test_indices]
            train_args_list.append(train_case_names)
            test_args_list.append(test_case_names)
        if return_case_ids:
            train_case_ids = case_ids[train_indices]
            test_case_ids = case_ids[test_indices]
            train_args_list.append(train_case_ids)
            test_args_list.append(test_case_ids)

        dset_train = tf.data.Dataset.from_tensor_slices(tuple(train_args_list))
        dset_test = tf.data.Dataset.from_tensor_slices(tuple(test_args_list))

        if batch_size is not None:
            dset_train = dset_train.batch(batch_size)
            dset_test = dset_test.batch(batch_size)

        return_list = return_list + [dset_train, dset_test]

    if return_case_ids:
        return_list.append(case_id_map)

    return return_list
        
    

if __name__ == '__main__':
    fname = '/storage/pyfr/meshes_maprun2/postprocessed.h5'
    sensor_masks = list(itertools.chain.from_iterable([[['p'] for _ in range(10)],[['u','v'] for _ in range(4)]]))
    full_field_mask = ['u','v']
    retain_time_dimension = False
    normalization = 'sensor_mean_center'
    ShallowDecoderDataset(fname, sensor_masks = sensor_masks, full_field_mask = full_field_mask, retain_time_dimension = retain_time_dimension, normalization = normalization)
    
    
    
    
    
